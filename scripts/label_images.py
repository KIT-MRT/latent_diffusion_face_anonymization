#!/usr/bin/env python3
"""Labeling tool for humans and license plates with click-to-draw bounding boxes.

Usage:
    python scripts/label_images.py

Requirements:
    - Images in data/test_images/
    - Output saved to data/test_labels.json
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np

DATA_DIR = Path("data")
TEST_IMAGES_DIR = DATA_DIR / "test_images"
LABELS_FILE = DATA_DIR / "test_labels.json"

drawing = False
start_x, start_y = -1, -1
boxes = []
current_class = "humans"


def draw_boxes(img, boxes_list, color, label):
    for box in boxes_list:
        x, y, w, h = box
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        cv2.putText(img, label, (int(x), int(y - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, boxes
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Labeling Tool", img_copy)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - start_x) > 5 and abs(y - start_y) > 5:
            boxes.append([start_x, start_y, x - start_x, y - start_y])


def load_existing_labels():
    if LABELS_FILE.exists():
        with open(LABELS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_labels(labels):
    DATA_DIR.mkdir(exist_ok=True)
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f, indent=2)


def get_image_files():
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    files = []
    for f in TEST_IMAGES_DIR.iterdir():
        if f.suffix.lower() in image_extensions:
            files.append(f)
    return sorted(files)


def label_image(image_path, existing_labels=None):
    global boxes, current_class
    existing = existing_labels.get(image_path.name, None)
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    original_img = img.copy()
    boxes = []
    current_class = "humans"
    
    print(f"\n=== Labeling: {image_path.name} ===")
    print(f"Instructions:")
    print(f"  - Click and drag to draw bounding boxes")
    print(f"  - Press 'h' to switch to HUMAN labeling")
    print(f"  - Press 'l' to switch to LICENSE PLATE labeling")
    print(f"  - Press 'u' to undo last box")
    print(f"  - Press 'n' when DONE with current class")
    print(f"  - Press 's' to SKIP this image")
    print(f"  - Press 'q' to QUIT")
    
    cv2.namedWindow("Labeling Tool")
    cv2.setMouseCallback("Labeling Tool", mouse_callback, img)
    
    human_boxes = []
    lp_boxes = []
    
    while True:
        display_img = original_img.copy()
        
        display_img = draw_boxes(display_img, human_boxes, (0, 255, 0), "Human")
        display_img = draw_boxes(display_img, lp_boxes, (255, 0, 0), "LP")
        display_img = draw_boxes(display_img, boxes, (0, 255, 255), current_class.upper())
        
        status_text = f"Mode: {current_class.upper()} | Humans: {len(human_boxes)} | LPs: {len(lp_boxes)}"
        cv2.putText(display_img, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Labeling Tool", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('h'):
            current_class = "humans"
            print("Switched to HUMAN labeling")
        elif key == ord('l'):
            current_class = "license_plates"
            print("Switched to LICENSE PLATE labeling")
        elif key == ord('u'):
            if boxes:
                boxes.pop()
                print(f"Undo. Current boxes: {len(boxes)}")
        elif key == ord('n'):
            if current_class == "humans":
                human_boxes = boxes.copy()
                boxes = []
                current_class = "license_plates"
                print(f"Done with humans. Now labeling license plates...")
            else:
                lp_boxes = boxes.copy()
                boxes = []
                break
        elif key == ord('s'):
            print("Skipping image...")
            cv2.destroyAllWindows()
            return None
        elif key == ord('q'):
            print("Quitting...")
            cv2.destroyAllWindows()
            exit()
    
    cv2.destroyAllWindows()
    
    h, w = img.shape[:2]
    
    human_data = []
    for i, box in enumerate(human_boxes):
        x, y, box_w, box_h = box
        human_data.append({
            "id": i + 1,
            "bbox": [float(x), float(y), float(x + box_w), float(y + box_h)],
            "height_px": float(abs(box_h)),
            "height_pct": float(abs(box_h) / h)
        })
    
    lp_data = []
    for i, box in enumerate(lp_boxes):
        x, y, box_w, box_h = box
        area_px = abs(box_w * box_h)
        area_pct = area_px / (w * h)
        lp_data.append({
            "id": i + 1,
            "bbox": [float(x), float(y), float(x + box_w), float(y + box_h)],
            "width_px": float(abs(box_w)),
            "height_px": float(abs(box_h)),
            "area_pct": float(area_pct)
        })
    
    result = {
        "humans": human_data,
        "license_plates": lp_data
    }
    
    print(f"\n--- Summary for {image_path.name} ---")
    print(f"Humans: {len(human_boxes)} bounding boxes")
    print(f"License plates: {len(lp_boxes)} bounding boxes")
    
    return result


def main():
    print("=" * 60)
    print("LABELING TOOL FOR HUMANS AND LICENSE PLATES")
    print("=" * 60)
    print(f"\nImages directory: {TEST_IMAGES_DIR}")
    print(f"Output file: {LABELS_FILE}")
    
    image_files = get_image_files()
    print(f"\nFound {len(image_files)} images to label")
    
    if not image_files:
        print("No images found in data/test_images/")
        return
    
    existing_labels = load_existing_labels()
    print(f"Found {len(existing_labels)} existing labels")
    
    for i, image_path in enumerate(image_files):
        print(f"\n{'='*60}")
        print(f"Image {i+1}/{len(image_files)}: {image_path.name}")
        print(f"{'='*60}")
        
        if image_path.name in existing_labels:
            print(f"Already labeled! Skipping...")
            continue
        
        result = label_image(image_path, existing_labels)
        
        if result and (result["humans"] or result["license_plates"]):
            existing_labels[image_path.name] = result
            save_labels(existing_labels)
            print(f"\nSaved labels for {image_path.name}")
        else:
            print(f"\nNo annotations for {image_path.name}, not saving")
    
    print("\n" + "=" * 60)
    print("LABELING COMPLETE!")
    print("=" * 60)
    print(f"Total images labeled: {len(existing_labels)}")
    print(f"Output saved to: {LABELS_FILE}")


if __name__ == "__main__":
    main()
