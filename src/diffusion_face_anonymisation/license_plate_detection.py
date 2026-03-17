"""License plate detection using Autolane YOLOX-S model (90.28% AP).

Note: Autolane was trained on US license plates. For German/EU plates,
lower confidence thresholds (0.15) and larger input sizes (800x800) help.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import torch

# Add YOLOX to path if installed from source
sys.path.insert(0, "/tmp/YOLOX")

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

from diffusion_face_anonymisation.license_plate import LicensePlate

# Default weights path
DEFAULT_WEIGHTS = "/home/roesch/tmp/latent_diffusion_face_anonymization/autolane_yolo_v8.pth"


class LicensePlateDetector:
    """License plate detection using Autolane YOLOX-S (90.28% AP).
    
    Note: Model trained on US plates. For EU/German plates, use lower
    confidence (0.15) and larger input size (800) for better recall.
    """

    def __init__(self, weights_path: str = None, conf_threshold: float = 0.15, 
                 input_size: int = 800):
        self.conf_threshold = conf_threshold
        self.nms_threshold = 0.45
        self.test_size = (input_size, input_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Find weights
        weights_path = weights_path or DEFAULT_WEIGHTS
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # Build YOLOX-S model (matches Autolane checkpoint architecture)
        exp = get_exp(None, "yolox-s")
        exp.num_classes = 1  # License plates only
        self.model = exp.get_model()

        # Load checkpoint
        ckpt = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval().to(self.device)

    def detect(self, img_path: Path) -> list[LicensePlate]:
        """Detect license plates in image. Returns list of LicensePlate objects."""
        img = cv2.imread(str(img_path))
        if img is None:
            return []

        # Preprocess
        img_prep, ratio = preproc(img, self.test_size)
        img_tensor = torch.from_numpy(img_prep).unsqueeze(0).float().to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = postprocess(
                outputs,
                num_classes=1,
                conf_thre=self.conf_threshold,
                nms_thre=self.nms_threshold,
            )

        if outputs[0] is None:
            return []

        # Convert to LicensePlate objects
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = outputs[0].cpu().numpy()
        detections[:, :4] /= ratio  # Scale back to original size

        plates = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            lp = LicensePlate(corners)
            lp.set_lp_cutout(img_rgb)
            plates.append(lp)

        return plates

    # Alias for compatibility with existing code
    def detect_lp_in_image(self, img_path: Path) -> list[LicensePlate]:
        return self.detect(img_path)
