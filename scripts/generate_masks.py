"""Generate Cityscapes-style segmentation masks using SAM3.

Detects bodies and license plates using SAM3 (facebook/sam3) and writes:
  - {stem}_labelIds.png    : uint8 class mask (0=bg, 1=body, 2=license_plate)
  - {stem}_instanceIds.png : uint8 instance mask (global instance id 1-255)
  - {stem}_overlay.png     : original image with colored semi-transparent masks

Pipeline:
  [Loader thread] -> load_queue -> [GPU main thread] -> save_queue -> [Saver workers x N]

Usage:
    python scripts/generate_masks.py \\
        --image_dir /path/to/images \\
        --output_dir /path/to/output \\
        --targets body lp \\
        --batch_size 4 \\
        --num_workers 4
"""

import argparse
import logging
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SENTINEL = object()

# Class definitions
CLASS_IDS = {
    "body": 1,
    "lp": 2,
}

CLASS_PROMPTS = {
    "body": "person",
    "lp": "license plate",
}

# Colors as (R, G, B)
CLASS_COLORS = {
    1: (0, 120, 255),    # body: blue
    2: (255, 180, 0),    # license plate: orange
}

OVERLAY_ALPHA = 100  # out of 255


class SAM3MaskExporter:
    """Loads SAM3 once and runs text-prompted segmentation for any class."""

    def __init__(self, threshold: float = 0.5, mask_threshold: float = 0.5, device: str = None):
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading SAM3 model on {self.device}...")
        from transformers import Sam3Model, Sam3Processor
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model.eval()
        logger.info("SAM3 loaded.")

    def preprocess(self, images: list[Image.Image], text_prompt: str):
        texts = [text_prompt] * len(images)
        return self.processor(images=images, text=texts, return_tensors="pt").to(self.device)

    def infer(self, inputs):
        with torch.no_grad():
            return self.model(**inputs)

    def postprocess(self, outputs, target_sizes: list[tuple[int, int]]) -> list[list[np.ndarray]]:
        batch_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )
        return [
            [m.cpu().numpy().astype(np.uint8) for m in result.get("masks", [])]
            for result in batch_results
        ]

    def detect(self, image: Image.Image, text_prompt: str) -> list[np.ndarray]:
        return self.postprocess(self.infer(self.preprocess([image], text_prompt)), [(image.height, image.width)])[0]


def build_masks_single(
    masks_by_class: dict[int, list[np.ndarray]],
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build labelIds and instanceIds arrays from pre-detected masks.

    Instance IDs are global across classes (1-255). labelIds is needed to
    recover the class of each instance.
    """
    label_arr = np.zeros((h, w), dtype=np.uint8)
    instance_arr = np.zeros((h, w), dtype=np.uint8)
    instance_id = 1
    for class_id, masks in masks_by_class.items():
        for mask in masks:
            if instance_id > 255:
                logger.warning("More than 255 instances in one image, skipping remaining.")
                break
            label_arr[mask == 1] = class_id
            instance_arr[mask == 1] = instance_id
            instance_id += 1
    return label_arr, instance_arr


def load_batch(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def gpu_infer_batch(
    detector: SAM3MaskExporter,
    images: list[Image.Image],
    targets: list[str],
) -> tuple[dict[str, object], list[tuple[int, int]]]:
    """CPU preprocessing + GPU forward pass for all targets. Returns raw outputs."""
    target_sizes = [(img.height, img.width) for img in images]
    raw_outputs = {}
    for target in targets:
        inputs = detector.preprocess(images, CLASS_PROMPTS[target])
        raw_outputs[target] = detector.infer(inputs)
    return raw_outputs, target_sizes


def postprocess_batch(
    detector: SAM3MaskExporter,
    raw_outputs: dict[str, object],
    target_sizes: list[tuple[int, int]],
    targets: list[str],
) -> list[dict[int, list[np.ndarray]]]:
    """Run CPU postprocessing for all targets. Returns per-image masks."""
    n = len(target_sizes)
    per_image: list[dict[int, list[np.ndarray]]] = [{} for _ in range(n)]
    for target in targets:
        class_id = CLASS_IDS[target]
        for i, masks in enumerate(detector.postprocess(raw_outputs[target], target_sizes)):
            per_image[i][class_id] = masks
    return per_image


def save_label_ids(label_arr: np.ndarray, path: Path):
    Image.fromarray(label_arr).save(path)


def save_instance_ids(instance_arr: np.ndarray, path: Path):
    Image.fromarray(instance_arr).save(path)


def save_overlay(image: Image.Image, masks_by_class: dict[int, list[np.ndarray]], path: Path):
    overlay = image.convert("RGBA")
    for class_id, masks in masks_by_class.items():
        color = CLASS_COLORS[class_id]
        for mask in masks:
            layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            layer.paste(Image.new("RGBA", overlay.size, (*color, OVERLAY_ALPHA)), mask=mask_pil)
            overlay = Image.alpha_composite(overlay, layer)
    overlay.convert("RGB").save(path)


def find_images(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in extensions)


def save_results(
    image_path: Path,
    image: Image.Image,
    masks_by_class: dict[int, list[np.ndarray]],
    image_dir: Path,
    output_dir: Path,
):
    rel = image_path.relative_to(image_dir)
    out_subdir = output_dir / rel.parent
    out_subdir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    label_arr, instance_arr = build_masks_single(masks_by_class, image.height, image.width)
    save_label_ids(label_arr, out_subdir / f"{stem}_labelIds.png")
    save_instance_ids(instance_arr, out_subdir / f"{stem}_instanceIds.png")
    save_overlay(image, masks_by_class, out_subdir / f"{stem}_overlay.png")


def _loader(batches: list[list[Path]], load_queue: queue.Queue):
    """Loads image batches from disk and feeds them into load_queue."""
    for batch_paths in batches:
        try:
            images = load_batch(batch_paths)
            load_queue.put((batch_paths, images))
        except Exception as e:
            logger.error(f"Failed to load batch starting at {batch_paths[0]}: {e}")
    load_queue.put(_SENTINEL)


def _saver(
    save_queue: queue.Queue,
    image_dir: Path,
    output_dir: Path,
    pbar: tqdm,
):
    """Saves processed results from save_queue to disk."""
    while True:
        item = save_queue.get()
        if item is _SENTINEL:
            break
        batch_paths, images, per_image_masks = item
        for image_path, image, masks_by_class in zip(batch_paths, images, per_image_masks):
            try:
                save_results(image_path, image, masks_by_class, image_dir, output_dir)
            except Exception as e:
                logger.error(f"Failed to save {image_path}: {e}")
            pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Cityscapes-style masks using SAM3."
    )
    parser.add_argument(
        "--image_dir", type=Path, required=True, help="Directory of input images."
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Directory for output masks."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["body", "lp"],
        default=["body", "lp"],
        help="Classes to detect (default: body lp).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="SAM3 detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="SAM3 mask binarization threshold (default: 0.5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cuda' or 'cpu' (auto-detected if omitted).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images per GPU forward pass (default: 4).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel file-writing threads (default: 4).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image_dir.is_dir():
        logger.error(f"--image_dir does not exist: {args.image_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(args.image_dir)
    if not image_paths:
        logger.error(f"No images found in {args.image_dir}")
        sys.exit(1)

    logger.info(
        f"Found {len(image_paths)} image(s). Targets: {args.targets}, "
        f"batch_size: {args.batch_size}, num_workers: {args.num_workers}"
    )

    detector = SAM3MaskExporter(
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        device=args.device,
    )

    batches = [image_paths[i:i + args.batch_size] for i in range(0, len(image_paths), args.batch_size)]

    load_queue: queue.Queue = queue.Queue(maxsize=4)
    save_queue: queue.Queue = queue.Queue(maxsize=args.num_workers * 2)

    with tqdm(total=len(image_paths), desc="Generating masks") as pbar:
        # Start loader thread
        threading.Thread(target=_loader, args=(batches, load_queue), daemon=True).start()

        # Start saver workers
        saver_threads = [
            threading.Thread(target=_saver, args=(save_queue, args.image_dir, args.output_dir, pbar), daemon=True)
            for _ in range(args.num_workers)
        ]
        for t in saver_threads:
            t.start()

        # GPU loop (main thread)
        while True:
            item = load_queue.get()
            if item is _SENTINEL:
                # Signal all savers to stop
                for _ in range(args.num_workers):
                    save_queue.put(_SENTINEL)
                break

            batch_paths, images = item
            try:
                raw_outputs, target_sizes = gpu_infer_batch(detector, images, args.targets)
                per_image_masks = postprocess_batch(detector, raw_outputs, target_sizes, args.targets)
                save_queue.put((batch_paths, images, per_image_masks))
            except Exception as e:
                logger.error(f"Detection failed for batch starting at {batch_paths[0]}: {e}")
                pbar.update(len(batch_paths))

        # Wait for all savers to finish
        for t in saver_threads:
            t.join()


if __name__ == "__main__":
    main()
