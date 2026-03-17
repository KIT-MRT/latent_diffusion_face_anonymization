import os
import os
os.environ['YOLO_VERBOSE'] = 'False'

import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

import diffusion_face_anonymisation.utils as dfa_utils
from diffusion_face_anonymisation.io_functions import (
    save_anon_image,
    get_bodies_from_file,
)
from diffusion_face_anonymisation.anonymization_functions import (
    define_anon_function,
    anonymize_body_image_with_cached_bodies,
)
from diffusion_face_anonymisation.body_detection import BodyDetector

# Set up logging - only WARNING and above to reduce spam
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("body_anon.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CUSTOM_TQDM_FORMAT = (
    "{l_bar}{bar}| "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] "
    "{postfix}"
)


def get_completed_images(output_dir: Path, anon_methods: list[str]) -> set:
    """Get set of image stems that have all methods completed."""
    if not output_dir.exists():
        return set()
    
    completed = set()
    for f in output_dir.glob("*_anon_white.png"):
        stem = f.stem.replace("_anon_white", "")
        has_all = True
        for method in anon_methods:
            method_file = output_dir / f"{stem}_anon_{method}.png"
            if not method_file.exists():
                has_all = False
                break
        if has_all:
            completed.add(stem)
    return completed


def check_progress(output_dir: Path, anon_methods: list[str], total_input_images: int):
    """Check and report processing progress."""
    print(f"\n{'='*70}")
    print(f"  📊 PROGRESS CHECK")
    print(f"{'='*70}")
    
    if not output_dir.exists():
        print(f"  Output directory does not exist yet.")
        return 0, set()
    
    # Count per method
    method_counts = {}
    for method in anon_methods:
        count = len(list(output_dir.glob(f"*_anon_{method}.png")))
        method_counts[method] = count
    
    # Get completed images (all methods)
    completed_images = get_completed_images(output_dir, anon_methods)
    
    print(f"  Input images:        {total_input_images}")
    print(f"{'─'*70}")
    print(f"  📈 By method:")
    for method, count in method_counts.items():
        pct = (count / total_input_images * 100) if total_input_images > 0 else 0
        print(f"     {method:8s}: {count:5d} / {total_input_images} ({pct:5.1f}%)")
    
    print(f"{'─'*70}")
    print(f"  ✅ Fully processed: {len(completed_images):5d} / {total_input_images}")
    
    remaining = total_input_images - len(completed_images)
    print(f"  ⏳ Remaining:        {remaining}")
    print(f"{'='*70}")
    
    return len(completed_images), completed_images


def get_parser():
    """Add custom arguments to the parser."""
    parser = argparse.ArgumentParser(
        prog="Body Anonymization",
        description="Anonymize bodies in images with parallel processing and batching.",
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_extension", type=str, default="png")
    parser.add_argument(
        "--anon_function",
        type=str,
        required=True,
        choices=["white", "gauss", "pixel", "lda", "all"],
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for anonymization",
    )
    parser.add_argument(
        "--parallel-lda",
        type=int,
        default=4,
        help="Number of parallel threads for LDA API calls per image",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for YOLO detection (higher = faster GPU utilization)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Enable streaming/pipelining mode (default: enabled)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable streaming mode - run detection then anonymization in separate phases",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip images that already have all output files generated",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        default=False,
        help="Only check what's been processed and exit (useful with --resume)",
    )
    return parser


def process_single_image(args):
    """Worker function to process a single image with all methods."""
    img_id, image_file, bodies, methods, output_dir, debug_dir = args
    
    results = []
    for method in methods:
        anon_function = define_anon_function(method)
        try:
            anon_img, _ = anonymize_body_image_with_cached_bodies(
                image_file, bodies, anon_function
            )
            save_anon_image(anon_img, image_file, output_dir, method)
            results.append((method, True, ""))
        except Exception as e:
            results.append((method, False, str(e)))
    
    return img_id, results


def run_streaming_pipeline(image_files, anon_methods, output_dir, debug_dir, 
                          batch_size=8, num_workers=4):
    """Run detection and anonymization in streaming/pipeline mode.
    
    Producer: Batch detects bodies with YOLO, puts results in queue
    Consumer: Waits for results, anonymizes when available
    """
    print(f"\nRunning in STREAMING mode (batch_size={batch_size}, workers={num_workers})")
    
    result_queue = queue.Queue(maxsize=50)
    completed_count = 0
    total_outputs = len(image_files) * len(anon_methods)
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    batches_processed = [0]
    start_time = time.time()
    
    method_counts = {m: 0 for m in anon_methods}
    method_lock = threading.Lock()
    
    pbar = tqdm(
        total=total_outputs, 
        desc="Processing", 
        unit="img", 
        ncols=120,
        bar_format=CUSTOM_TQDM_FORMAT,
        dynamic_ncols=True
    )
    
    def producer():
        """Producer: batch detection with YOLO"""
        detector = BodyDetector(batch_size=batch_size)
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_files))
            batch = image_files[start_idx:end_idx]
            
            try:
                bodies_dict = detector.batch_detect(batch)
                for img_file, bodies in bodies_dict.items():
                    result_queue.put((img_file, bodies))
            except Exception as e:
                logger.error(f"Batch detection error: {e}")
                for img_file in batch:
                    result_queue.put((img_file, []))
            
            batches_processed[0] = batch_idx + 1
        
        # Signal completion
        for _ in range(num_workers):
            result_queue.put(None)
    
    def consumer(worker_id):
        """Consumer: anonymization worker"""
        nonlocal completed_count
        
        while True:
            try:
                item = result_queue.get(timeout=2)
            except queue.Empty:
                continue
            
            if item is None:
                try:
                    result_queue.put(None)
                except:
                    pass
                break
            
            image_file, bodies = item
            
            for method in anon_methods:
                anon_function = define_anon_function(method)
                try:
                    anon_img, _ = anonymize_body_image_with_cached_bodies(
                        image_file, bodies, anon_function
                    )
                    save_anon_image(anon_img, image_file, output_dir, method)
                    
                    with method_lock:
                        method_counts[method] += 1
                        completed_count += 1
                    
                    pbar.update(1)
                    
                    elapsed = time.time() - start_time
                    speed = completed_count / elapsed if elapsed > 0 else 0
                    eta = (total_outputs - completed_count) / speed if speed > 0 else 0
                    queue_size = result_queue.qsize()
                    
                    counts_str = " | ".join([f"{m}:{method_counts[m]}" for m in anon_methods])
                    
                    pbar.set_postfix({
                        'det': f'{batches_processed[0]}/{total_batches}',
                        'queue': queue_size,
                        'counts': counts_str,
                        'total': completed_count,
                        'speed': f'{speed:.1f}/s',
                        'eta': f'{eta/60:.1f}m'
                    })
                except Exception as e:
                    logger.error(f"Error: {e}")
                    with method_lock:
                        completed_count += 1
                    pbar.update(1)
    
    # Start producer
    producer_thread = threading.Thread(target=producer)
    producer_thread.start()
    
    # Start consumers
    consumer_threads = [threading.Thread(target=consumer, args=(i,)) for i in range(num_workers)]
    for t in consumer_threads:
        t.start()
    
    # Wait
    producer_thread.join()
    for t in consumer_threads:
        t.join()
    
    pbar.close()
    return completed_count


def run_batch_mode(image_files, anon_methods, output_dir, debug_dir, batch_size=8, num_workers=4):
    """Run detection then anonymization in two separate phases (legacy mode)."""
    print(f"\nRunning in BATCH mode (batch_size={batch_size}, workers={num_workers})")
    
    total_outputs = len(image_files) * len(anon_methods)
    method_counts = {m: 0 for m in anon_methods}
    
    # Phase 1: Batch detection
    print("Phase 1: Batch detection...")
    detector = BodyDetector(batch_size=batch_size)
    
    cached_bodies = {}
    pbar_detect = tqdm(
        total=len(image_files), 
        desc="Detecting", 
        unit="img",
        bar_format=CUSTOM_TQDM_FORMAT,
        dynamic_ncols=True
    )
    start_time = time.time()
    
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]
        try:
            bodies_dict = detector.batch_detect(batch)
            for img_file, bodies in bodies_dict.items():
                cached_bodies[img_file] = bodies
                pbar_detect.update(1)
        except Exception as e:
            logger.error(f"Error in batch: {e}")
            pbar_detect.update(len(batch))
    
    pbar_detect.close()
    detect_time = time.time() - start_time
    print(f"Detection: {detect_time:.1f}s ({detect_time/len(image_files):.2f}s/img)")
    
    # Phase 2: Parallel anonymization
    print(f"\nPhase 2: Anonymizing...")
    
    work_items = [
        (img_id, img_file, bodies, anon_methods, output_dir, debug_dir)
        for img_id, (img_file, bodies) in enumerate(cached_bodies.items())
    ]
    
    start_time = time.time()
    completed = 0
    pbar_anon = tqdm(
        total=total_outputs, 
        desc="Anonymizing", 
        unit="img",
        bar_format=CUSTOM_TQDM_FORMAT,
        dynamic_ncols=True
    )
    
    def process_single_image_with_count(args):
        """Worker function that updates method counts."""
        img_id, image_file, bodies, methods, output_dir, debug_dir = args
        
        results = []
        for method in methods:
            anon_function = define_anon_function(method)
            try:
                anon_img, _ = anonymize_body_image_with_cached_bodies(
                    image_file, bodies, anon_function
                )
                save_anon_image(anon_img, image_file, output_dir, method)
                method_counts[method] += 1
                results.append((method, True, ""))
            except Exception as e:
                results.append((method, False, str(e)))
        
        return img_id, results
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image_with_count, item): item[0] for item in work_items}
        
        for future in as_completed(futures):
            try:
                img_id, results = future.result()
                completed += len(results)
                pbar_anon.update(len(results))
                
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (total_outputs - completed) / speed if speed > 0 else 0
                
                counts_str = " | ".join([f"{m}:{method_counts[m]}" for m in anon_methods])
                
                pbar_anon.set_postfix({
                    'counts': counts_str,
                    'total': completed,
                    'speed': f'{speed:.1f}/s',
                    'eta': f'{eta/60:.1f}m'
                })
            except Exception as e:
                logger.error(f"Error: {e}")
                completed += len(anon_methods)
                pbar_anon.update(len(anon_methods))
    
    pbar_anon.close()
    
    return completed


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Determine if we need detection (no mask_dir provided)
    detect = not args.mask_dir
    
    # Extract arguments
    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir) if args.mask_dir else Path("")
    output_dir = Path(args.output_dir)
    anon_method = args.anon_function
    image_extension = args.image_extension
    num_workers = args.num_workers
    parallel_lda = args.parallel_lda
    batch_size = args.batch_size
    streaming = args.streaming
    
    # Define methods to run
    if anon_method == "all":
        anon_methods = ["white", "gauss", "pixel", "lda"]
    else:
        anon_methods = [anon_method]
    
    print(f"\n{'='*70}")
    print(f"  Body Anonymization - Optimized Pipeline")
    print(f"{'='*70}")
    print(f"  📁 Input:    {args.image_dir}")
    print(f"  📁 Output:   {args.output_dir}")
    print(f"  ⚙️  Mode:    {'Streaming' if streaming else 'Batch'}")
    print(f"  📦 Batch:    {batch_size} | 👷 Workers: {num_workers}")
    if args.resume:
        print(f"  🔄 Resume:   Enabled (skip completed)")
    if args.check_only:
        print(f"  🔍 Check:   Only showing progress")
    print(f"{'─'*70}")
    print(f"  🔧 Methods to apply:")
    for m in anon_methods:
        print(f"     • {m}")
    print(f"{'─'*70}")
    print(f"  📊 Expected outputs:")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    debug_dir = Path(os.path.join(output_dir, "debug"))
    os.makedirs(debug_dir, exist_ok=True)
    
    # Get all images
    image_mask_dict = dfa_utils.get_image_mask_dict(
        str(image_dir),
        str(mask_dir),
        method="body",
        image_file_extension=image_extension,
        detect=detect,
    )
    
    # Extract image files (skip entries without image_file)
    image_files = []
    for entry in image_mask_dict.values():
        if 'image_file' in entry:
            image_files.append(entry['image_file'])
    
    total_input_images = len(image_files)
    
    # Check progress and optionally filter out completed images
    completed_count, completed_images = check_progress(
        output_dir, anon_methods, total_input_images
    )
    
    if args.check_only:
        print("\n✅ --check-only specified, exiting.")
        return
    
    # Filter out already completed images if --resume is specified
    if args.resume and completed_images:
        original_count = len(image_files)
        image_files = [
            f for f in image_files 
            if f.stem not in completed_images
        ]
        skipped = original_count - len(image_files)
        print(f"\n  🔄 --resume enabled: skipping {skipped} already-processed images")
        print(f"     Processing {len(image_files)} remaining images")
    
    total_images = len(image_files)
    total_outputs = total_images * len(anon_methods)
    
    print(f"     • {total_images} input images")
    for m in anon_methods:
        print(f"     • {total_images} {m} anonymized")
    print(f"     = {total_outputs} total outputs")
    print(f"{'='*70}\n")
    
    overall_start = time.time()
    
    if streaming:
        completed = run_streaming_pipeline(
            image_files, anon_methods, output_dir, debug_dir,
            batch_size=batch_size, num_workers=num_workers
        )
    else:
        completed = run_batch_mode(
            image_files, anon_methods, output_dir, debug_dir,
            batch_size=batch_size, num_workers=num_workers
        )
    
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*70}")
    print(f"  ✅ PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  ⏱️  Total time:    {overall_time/60:.1f} minutes ({overall_time:.1f}s)")
    print(f"  🚀 Throughput:   {completed/overall_time:.2f} images/second")
    print(f"  📈 Avg per img:  {overall_time/completed*1000:.0f} ms")
    print(f"{'='*70}")
    
    # Final statistics check
    print(f"\n{'='*70}")
    print(f"  FINAL STATISTICS CHECK")
    print(f"{'='*70}")
    
    output_files = list(output_dir.glob("*.png"))
    method_counts = {}
    for method in anon_methods:
        count = sum(1 for f in output_files if f"_anon_{method}.png" in f.name)
        method_counts[method] = count
    
    print(f"  Expected per method:  {total_images}")
    print(f"  Total outputs:         {len(output_files)}")
    print(f"{'─'*70}")
    print(f"  📊 By method:")
    all_complete = True
    for method, count in method_counts.items():
        status = "✅" if count == total_images else "❌"
        pct = (count / total_images * 100) if total_images > 0 else 0
        print(f"     {method:8s}: {count:4d} / {total_images} ({pct:5.1f}%) {status}")
        if count != total_images:
            all_complete = False
    
    missing = total_outputs - len(output_files)
    if missing > 0:
        print(f"\n  ⚠️  Missing outputs: {missing}")
    else:
        print(f"\n  ✅ All {total_outputs} outputs completed successfully!")
    
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
