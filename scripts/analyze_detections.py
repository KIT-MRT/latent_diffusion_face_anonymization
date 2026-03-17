#!/usr/bin/env python3
"""Detailed analysis of detection results.

This script analyzes:
1. Size-based performance (small vs large objects)
2. Per-image breakdown
3. Detection quality by size category
4. Error analysis
"""

import json
from pathlib import Path
import numpy as np

DATA_DIR = Path("data")
LABELS_FILE = DATA_DIR / "test_labels.json"
DETECTIONS_DIR = DATA_DIR / "detections"


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def categorize_by_size(gt_objects, class_name):
    """Categorize objects by size."""
    categories = {"small": [], "medium": [], "large": []}
    
    for obj in gt_objects:
        if class_name == "humans":
            size_metric = obj.get("height_pct", 0)
            if size_metric < 0.20:
                categories["small"].append(obj)
            elif size_metric < 0.40:
                categories["medium"].append(obj)
            else:
                categories["large"].append(obj)
        else:  # license_plates
            area_pct = obj.get("area_pct", 0)
            if area_pct < 0.0008:
                categories["small"].append(obj)
            elif area_pct < 0.002:
                categories["medium"].append(obj)
            else:
                categories["large"].append(obj)
    
    return categories


def analyze_by_size(gt, existing, sam3, class_name):
    """Analyze detection performance by object size."""
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS FOR {class_name.upper()}")
    print(f"{'='*60}")
    
    # Aggregate all gt objects across all images
    all_gt_objects = []
    for img_name in gt.keys():
        all_gt_objects.extend(gt[img_name].get(class_name, []))
    
    gt_cats = categorize_by_size(all_gt_objects, class_name)
    
    results = {}
    
    print(f"\n  GT objects by size:")
    for cat_name in ["small", "medium", "large"]:
        gt_objects = gt_cats[cat_name]
        print(f"    {cat_name}: {len(gt_objects)}")
    
    # Aggregate all detections across all images
    all_existing_dets = []
    all_sam3_dets = []
    for img_name in existing.keys():
        all_existing_dets.extend(existing[img_name].get(class_name, []))
    for img_name in sam3.keys():
        all_sam3_dets.extend(sam3[img_name].get(class_name, []))
    
    for cat_name in ["small", "medium", "large"]:
        gt_objects = gt_cats[cat_name]
        if not gt_objects:
            continue
        
        existing_matched = 0
        for det in all_existing_dets:
            for gt_obj in gt_objects:
                if compute_iou(det["bbox"], gt_obj["bbox"]) >= 0.3:
                    existing_matched += 1
                    break
        
        sam3_matched = 0
        for det in all_sam3_dets:
            for gt_obj in gt_objects:
                if compute_iou(det["bbox"], gt_obj["bbox"]) >= 0.3:
                    sam3_matched += 1
                    break
        
        gt_count = len(gt_objects)
        
        print(f"\n--- {cat_name.upper()} {class_name} (n={gt_count}) ---")
        print(f"  Size range: ", end="")
        if class_name == "humans":
            sizes = [f"{o['height_pct']*100:.1f}%" for o in gt_objects]
            print(f"height {min(sizes)} - {max(sizes)}")
        else:
            sizes = [f"{o['area_pct']*100:.3f}%" for o in gt_objects]
            print(f"area {min(sizes)} - {max(sizes)}")
        
        existing_recall = existing_matched / gt_count if gt_count > 0 else 0
        sam3_recall = sam3_matched / gt_count if gt_count > 0 else 0
        
        print(f"  Existing recall: {existing_matched}/{gt_count} = {existing_recall*100:.1f}%")
        print(f"  SAM3 recall:     {sam3_matched}/{gt_count} = {sam3_recall*100:.1f}%")
        
        results[cat_name] = {
            "gt_count": gt_count,
            "existing_matched": existing_matched,
            "sam3_matched": sam3_matched,
            "existing_recall": existing_recall,
            "sam3_recall": sam3_recall
        }
    
    return results


def per_image_analysis(gt, existing, sam3):
    """Show per-image breakdown."""
    print(f"\n{'='*60}")
    print("PER-IMAGE BREAKDOWN")
    print(f"{'='*60}")
    
    print(f"\n{'Image':<50} {'Class':<10} {'GT':<4} {'Existing':<10} {'SAM3':<10}")
    print("-" * 90)
    
    for img_name in sorted(gt.keys()):
        gt_data = gt[img_name]
        existing_data = existing.get(img_name, {"humans": [], "license_plates": []})
        sam3_data = sam3.get(img_name, {"humans": [], "license_plates": []})
        
        for class_name in ["humans", "license_plates"]:
            gt_count = len(gt_data.get(class_name, []))
            existing_count = len(existing_data.get(class_name, []))
            sam3_count = len(sam3_data.get(class_name, []))
            
            short_name = img_name[:47] + "..." if len(img_name) > 50 else img_name
            print(f"{short_name:<50} {class_name:<10} {gt_count:<4} {existing_count:<10} {sam3_count:<10}")


def error_analysis(gt, existing, sam3, class_name):
    """Analyze specific errors."""
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS FOR {class_name.upper()}")
    print(f"{'='*60}")
    
    for img_name in sorted(gt.keys()):
        gt_objects = gt[img_name].get(class_name, [])
        if not gt_objects:
            continue
        
        existing_dets = existing.get(img_name, {}).get(class_name, [])
        sam3_dets = sam3.get(img_name, {}).get(class_name, [])
        
        existing_matched = []
        sam3_matched = []
        
        for i, det in enumerate(existing_dets):
            for j, gt_obj in enumerate(gt_objects):
                if compute_iou(det["bbox"], gt_obj["bbox"]) >= 0.3:
                    existing_matched.append(j)
                    break
        
        for i, det in enumerate(sam3_dets):
            for j, gt_obj in enumerate(gt_objects):
                if compute_iou(det["bbox"], gt_obj["bbox"]) >= 0.3:
                    sam3_matched.append(j)
                    break
        
        existing_missed = [j for j in range(len(gt_objects)) if j not in existing_matched]
        sam3_missed = [j for j in range(len(gt_objects)) if j not in sam3_matched]
        
        if existing_missed or sam3_missed:
            print(f"\n{img_name}")
            print(f"  GT: {len(gt_objects)} {class_name}")
            if existing_missed:
                sizes = [gt_objects[i].get("height_pct" if class_name == "humans" else "area_pct", 0) for i in existing_missed]
                print(f"  Existing MISSED: indices {existing_missed} (sizes: {sizes})")
            if sam3_missed:
                sizes = [gt_objects[i].get("height_pct" if class_name == "humans" else "area_pct", 0) for i in sam3_missed]
                print(f"  SAM3 MISSED: indices {sam3_missed} (sizes: {sizes})")


def main():
    print("="*60)
    print("DETAILED DETECTION ANALYSIS")
    print("="*60)
    
    gt = load_json(LABELS_FILE)
    existing = load_json(DETECTIONS_DIR / "existing" / "detections.json")
    sam3 = load_json(DETECTIONS_DIR / "sam3" / "detections.json")
    
    # Summary stats
    total_humans_gt = sum(len(v.get("humans", [])) for v in gt.values())
    total_lp_gt = sum(len(v.get("license_plates", [])) for v in gt.values())
    total_humans_existing = sum(len(v.get("humans", [])) for v in existing.values())
    total_lp_existing = sum(len(v.get("license_plates", [])) for v in existing.values())
    total_humans_sam3 = sum(len(v.get("humans", [])) for v in sam3.values())
    total_lp_sam3 = sum(len(v.get("license_plates", [])) for v in sam3.values())
    
    print(f"\n--- OVERALL TOTALS ---")
    print(f"Ground Truth:    {total_humans_gt} humans, {total_lp_gt} license plates")
    print(f"Existing Detected: {total_humans_existing} humans, {total_lp_existing} license plates")
    print(f"SAM3 Detected:     {total_humans_sam3} humans, {total_lp_sam3} license plates")
    
    # Size-based analysis
    analyze_by_size(gt, existing, sam3, "humans")
    analyze_by_size(gt, existing, sam3, "license_plates")
    
    # Per-image breakdown
    per_image_analysis(gt, existing, sam3)
    
    # Error analysis
    error_analysis(gt, existing, sam3, "humans")
    error_analysis(gt, existing, sam3, "license_plates")
    
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
