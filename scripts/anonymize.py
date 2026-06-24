#!/usr/bin/env python3
"""Unified anonymization script.

Usage:
    python anonymize.py --image_dir /data/images --output_dir /data/output \
        --targets body lp --method pixel

    python anonymize.py --image_dir /data --output_dir /out \
        --targets body lp --body_method lda --lp_method pixel

    # With config file (recommended for multi-GPU):
    python anonymize.py --config configs/config-everything.yaml

    # Or with CLI args:
    python anonymize.py --image_dir /data --output_dir /out \
        --targets body --body_method lda --gpu_count 4 --base_port 7860
"""

import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path

from diffusion_face_anonymisation.pipeline import (
    create_detector,
    get_image_files,
    run_streaming_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified anonymization")
    parser.add_argument("--config", type=str, help="Config YAML file")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--targets", type=str, nargs="+", default=["body", "lp"],
                       choices=["body", "lp"])
    parser.add_argument("--detector", type=str, default="sam3",
                       choices=["sam3", "yolo"])
    parser.add_argument("--method", type=str, default="white",
                       choices=["white", "gauss", "pixel"],
                       help="Method for all targets (can be overridden per target)")
    parser.add_argument("--body_method", type=str,
                       choices=["white", "gauss", "pixel", "lda"],
                       help="Override method for bodies")
    parser.add_argument("--lp_method", type=str,
                       choices=["white", "gauss", "pixel"],
                       help="Override method for license plates")
    parser.add_argument("--body_threshold", type=float, default=0.5)
    parser.add_argument("--lp_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ext", type=str, default="png")
    parser.add_argument("--resume", action="store_true")
    
    # GPU config (config file preferred)
    parser.add_argument("--gpu_count", type=int, help="Number of GPUs for LDA (overrides config)")
    parser.add_argument("--base_port", type=int, help="Base port for API endpoints (overrides config)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load config from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
    
    # Merge CLI args with config (CLI takes precedence)
    image_dir = args.image_dir or config.get('paths', {}).get('image_dir') or config.get('image_dir')
    output_dir = args.output_dir or config.get('paths', {}).get('output_dir') or config.get('output_dir')
    
    # Get anonymization config from config file
    anon_config = config.get('anonymization', {})
    
    targets = args.targets if args.targets != ['body', 'lp'] else anon_config.get('targets', ['body', 'lp'])
    detector = args.detector if args.config is None else anon_config.get('detector', args.detector)
    
    # Get methods - CLI overrides config
    methods_config = anon_config.get('methods', {})
    
    # Default method (from CLI or config)
    default_method = args.method
    
    # Per-target methods (CLI overrides config)
    body_method = args.body_method or methods_config.get('body', default_method)
    lp_method = args.lp_method or methods_config.get('lp', default_method)
    
    body_threshold = args.body_threshold if args.config is None else anon_config.get('thresholds', {}).get('body', args.body_threshold)
    lp_threshold = args.lp_threshold if args.config is None else anon_config.get('thresholds', {}).get('lp', args.lp_threshold)
    batch_size = args.batch_size if args.config is None else anon_config.get('batch_size', args.batch_size)
    ext = args.ext if args.config is None else anon_config.get('image_extension', args.ext)
    num_workers = args.num_workers if args.config is None else anon_config.get('num_workers', args.num_workers)
    
    # GPU config - CLI overrides config
    gpu_config = config.get('gpu', {})
    gpu_count = args.gpu_count or gpu_config.get('count')
    base_port = args.base_port or gpu_config.get('base_port', 7860)
    api_timeout = config.get('api', {}).get('timeout', 120)
    
    if not image_dir or not output_dir:
        print("Error: --image_dir and --output_dir are required (or set in config)")
        return
    
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images
    images = get_image_files(image_dir, ext)
    print(f"Found {len(images)} images")
    print(f"Targets: {targets}")
    print(f"Detector: {detector}")
    print(f"Methods: body={body_method}, lp={lp_method}")
    
    if gpu_count:
        print(f"Multi-GPU: {gpu_count} GPUs, base_port={base_port}")
    print(f"Workers: {num_workers}")
    
    # Build method dict for pipeline
    methods = {}
    if 'body' in targets:
        methods['body'] = body_method
    if 'lp' in targets:
        methods['lp'] = lp_method
    
    # Check progress (skip if resume)
    if args.resume:
        targets_str = "+".join(sorted(targets))
        
        if body_method == lp_method:
            pattern = f"*_anon_{targets_str}_{body_method}.png"
            completed = {f.stem.replace(f"_anon_{targets_str}_{body_method}", "") 
                        for f in output_dir.glob(pattern)}
        else:
            pattern1 = f"*_anon_{targets_str}_{body_method}_{lp_method}.png"
            pattern2 = f"*_anon_{targets_str}_{lp_method}_{body_method}.png"
            completed = {f.stem.replace(f"_anon_{targets_str}_{body_method}_{lp_method}", "")
                        for f in output_dir.glob(pattern1)}
            completed |= {f.stem.replace(f"_anon_{targets_str}_{lp_method}_{body_method}", "")
                         for f in output_dir.glob(pattern2)}
        
        images = [img for img in images if img.stem not in completed]
        skipped = len(get_image_files(image_dir, ext)) - len(images)
        print(f"Skipped {skipped} already-processed images")
    
    if not images:
        print("No images to process!")
        return
    
    # Create detectors
    detectors = {}
    for target in targets:
        if target == "body":
            detectors[target] = create_detector(
                "body", detector, 
                threshold=body_threshold, 
                batch_size=batch_size
            )
        elif target == "lp":
            detectors[target] = create_detector(
                "lp", detector,
                threshold=lp_threshold
            )
    
    # Create endpoint pool if LDA is used with GPU count
    endpoint_pool = None
    if gpu_count and 'lda' in methods.values():
        from diffusion_face_anonymisation.multi_gpu_utils import APIEndpointPool
        endpoint_pool = APIEndpointPool(
            base_port=base_port,
            gpu_count=gpu_count,
            timeout=api_timeout
        )
        print(f"Created API pool with {gpu_count} endpoints")
    
    # Run pipeline
    stats = run_streaming_pipeline(
        image_files=images,
        targets=targets,
        detectors=detectors,
        methods=methods,
        output_dir=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        endpoint_pool=endpoint_pool,
    )
    
    # Close endpoint pool if created
    if endpoint_pool:
        endpoint_pool.close()
    
    # Save stats
    stats_file = output_dir / "stats.json"
    output_data = {
        "input_dir": str(image_dir),
        "output_dir": str(output_dir),
        "targets": targets,
        "detector": detector,
        "methods": methods,
        "gpu_count": gpu_count,
        "base_port": base_port if gpu_count else None,
        **stats,
        "completed_at": datetime.now().isoformat()
    }
    with open(stats_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDone! Stats saved to {stats_file}")
    print(f"Processed {stats['total_outputs']} images in {stats['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
