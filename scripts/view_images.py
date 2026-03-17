#!/usr/bin/env python3
"""Simple viewer that saves images to a folder for external viewing.

Usage:
    python scripts/view_images.py
"""

import cv2
from pathlib import Path

TEST_IMAGES_DIR = Path("data/test_images")
OUTPUT_DIR = Path("data/view_for_labeling")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    images = sorted(TEST_IMAGES_DIR.glob("*.png"))
    
    print(f"Found {len(images)} images")
    print(f"Saving to {OUTPUT_DIR}/ for viewing...")
    
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is not None:
            # Resize for easier viewing
            scale = min(1.0, 1200 / img.shape[1])
            if scale < 1:
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            output_path = OUTPUT_DIR / f"{i:02d}_{img_path.name}"
            cv2.imwrite(str(output_path), img)
            print(f"Saved: {output_path.name}")
    
    print(f"\nAll images saved to {OUTPUT_DIR}/")
    print("You can view these images and tell me what you see.")

if __name__ == "__main__":
    main()
