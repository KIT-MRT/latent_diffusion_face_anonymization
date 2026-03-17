#!/usr/bin/env python3
"""Compare detection results and generate visualizations.

This script:
1. Loads ground truth labels
2. Loads detection results from existing and SAM3 detectors
3. Computes detection metrics (precision, recall, detection rate)
4. Generates visualization images with bounding boxes
5. Outputs comparison statistics

Usage:
    python scripts/compare_detections.py
"""

import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import cv2

DATA_DIR = Path("data")
TEST_IMAGES_DIR = DATA_DIR / "test_images"
DETECTIONS_DIR = DATA_DIR / "detections"
LABELS_FILE = DATA_DIR / "test_labels.json"
VIS_DIR = DATA_DIR / "visualizations"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.3):
    """Match detections to ground truth boxes and compute metrics.
    
    Args:
        detections: List of detected boxes
        ground_truth: List of ground truth boxes
        iou_threshold: Minimum IoU to consider a match
        
    Returns:
        Dictionary with matched, false_positives, missed counts
    """
    gt_matched = [False] * len(ground_truth)
    tp = 0
    fp = 0
    
    for det in detections:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(ground_truth):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou(det["bbox"], gt_box["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = sum(1 for matched in gt_matched if not matched)
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }


def compute_detection_metrics(detections, ground_truth, class_name):
    """Compute detection metrics for a class."""
    if class_name == "humans":
        det_list = detections.get("humans", [])
        gt_list = ground_truth.get("humans", [])
    else:
        det_list = detections.get("license_plates", [])
        gt_list = ground_truth.get("license_plates", [])
    
    metrics = match_detections_to_ground_truth(det_list, gt_list)
    metrics["detected_count"] = len(det_list)
    metrics["ground_truth_count"] = len(gt_list)
    metrics["detection_rate"] = len(gt_list) > 0 and len(det_list) / len(gt_list) or 0
    
    return metrics


def draw_boxes(ax, boxes, color, label, alpha=0.3):
    """Draw bounding boxes on matplotlib axis."""
    legend_patches = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor=color, 
                         facecolor=color, alpha=alpha)
        ax.add_patch(rect)
    
    if boxes:
        legend_patches.append(Patch(facecolor=color, edgecolor=color, 
                                     alpha=alpha, label=label))
    return legend_patches


def visualize_comparison(image_name, image_path, ground_truth, existing_results, sam3_results):
    """Create a visualization comparing all detection methods."""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    gt_color = "green"
    existing_color = "blue"
    sam3_color = "red"
    
    for ax, (title, detections) in zip(axes, [
        ("Ground Truth", ground_truth),
        ("Existing (YOLO + YOLOX)", existing_results),
        ("SAM3", sam3_results)
    ]):
        ax.imshow(img_array)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        
        patches = []
        
        if title == "Ground Truth":
            patches += draw_boxes(ax, ground_truth.get("humans", []), gt_color, "Human (GT)")
            patches += draw_boxes(ax, ground_truth.get("license_plates", []), gt_color, "LP (GT)")
        else:
            patches += draw_boxes(ax, detections.get("humans", []), 
                                existing_color if title == "Existing (YOLO + YOLOX)" else sam3_color, 
                                "Human")
            patches += draw_boxes(ax, detections.get("license_plates", []), 
                                existing_color if title == "Existing (YOLO + YOLOX)" else sam3_color, 
                                "LP")
        
        if patches:
            ax.legend(handles=patches, loc="upper right", fontsize=8)
    
    plt.suptitle(f"Comparison: {image_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = VIS_DIR / f"{image_name.replace('.png', '')}_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    print("=" * 60)
    print("DETECTION COMPARISON")
    print("=" * 60)
    
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading ground truth labels...")
    ground_truth = load_json(LABELS_FILE)
    print(f"Loaded {len(ground_truth)} ground truth labels")
    
    logger.info("Loading existing detector results...")
    existing_results = load_json(DETECTIONS_DIR / "existing" / "detections.json")
    print(f"Loaded {len(existing_results)} existing detector results")
    
    logger.info("Loading SAM3 detector results...")
    sam3_results = load_json(DETECTIONS_DIR / "sam3" / "detections.json")
    print(f"Loaded {len(sam3_results)} SAM3 detector results")
    
    print("\n--- Per-Class Metrics ---\n")
    
    all_metrics = {
        "humans": {"existing": [], "sam3": []},
        "license_plates": {"existing": [], "sam3": []}
    }
    
    for image_name in ground_truth:
        gt = ground_truth[image_name]
        existing = existing_results.get(image_name, {"humans": [], "license_plates": []})
        sam3 = sam3_results.get(image_name, {"humans": [], "license_plates": []})
        
        for class_name in ["humans", "license_plates"]:
            existing_metrics = compute_detection_metrics(existing, gt, class_name)
            sam3_metrics = compute_detection_metrics(sam3, gt, class_name)
            
            all_metrics[class_name]["existing"].append(existing_metrics)
            all_metrics[class_name]["sam3"].append(sam3_metrics)
        
        vis_path = visualize_comparison(
            image_name,
            TEST_IMAGES_DIR / image_name,
            gt, existing, sam3
        )
        logger.info(f"Saved visualization: {vis_path}")
    
    print("\n" + "=" * 80)
    print(f"{'Class':<20} {'Method':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("=" * 80)
    
    summary = {}
    
    for class_name in ["humans", "license_plates"]:
        for method, method_key in [("Existing", "existing"), ("SAM3", "sam3")]:
            metrics_list = all_metrics[class_name][method_key]
            
            total_tp = sum(m["true_positives"] for m in metrics_list)
            total_fp = sum(m["false_positives"] for m in metrics_list)
            total_fn = sum(m["false_negatives"] for m in metrics_list)
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0
            
            summary[class_name] = summary.get(class_name, {})
            summary[class_name][method] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn
            }
            
            print(f"{class_name:<20} {method:<15} {total_tp:<6} {total_fp:<6} {total_fn:<6} "
                  f"{precision:<12.3f} {recall:<12.3f} {f1:<10.3f}")
    
    print("=" * 80)
    
    summary_file = DATA_DIR / "detection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")
    
    print(f"\nVisualizations saved to: {VIS_DIR}/")
    print(f"Total images processed: {len(ground_truth)}")


if __name__ == "__main__":
    main()
