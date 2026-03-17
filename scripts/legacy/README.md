# Legacy Anonymization Scripts

**⚠️ DEPRECATED:** These scripts are maintained for backward compatibility only.

**New users should use `scripts/anonymize.py` instead.**

---

## Why Consolidate?

The new unified `anonymize.py` script provides:

- ✅ **Single entry point** for all anonymization tasks (face, body, license plates)
- ✅ **Better performance** with optimized streaming pipeline
- ✅ **Consistent output naming** across all methods
- ✅ **SAM3 support** with significantly higher detection accuracy (100% LP recall vs 35% for YOLOX)
- ✅ **Flexible target & method selection** - anonymize any combination with any method
- ✅ **Progress tracking & resume** - skip already processed images
- ✅ **Statistics output** - detailed JSON stats for analysis
- ✅ **Easier maintenance** - one codebase to improve and fix

---

## Migration Guide

### Face Anonymization

**Old command:**
```bash
python scripts/face_anonymization.py \
    --image_dir /data/images \
    --mask_dir /data/masks \
    --output_dir /data/output \
    --anon_function lda
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets face \
    --mask_dir /data/masks \
    --method lda
```

---

### Body Anonymization (YOLO)

**Old command:**
```bash
python scripts/body_anonymization.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --anon_function pixel \
    --batch-size 8 \
    --num-workers 4
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets body \
    --detector yolo \
    --method pixel \
    --batch_size 8 \
    --num_workers 4
```

---

### Body Anonymization (SAM3)

**Old command:**
```bash
python scripts/body_anonymization_sam3.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --body_anon pixel \
    --body_threshold 0.5
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets body \
    --method pixel \
    --body_threshold 0.5
# Note: SAM3 is now the default detector!
```

---

### License Plate Anonymization (YOLOX)

**Old command:**
```bash
python scripts/license_plate_anonymization.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --method white \
    --conf 0.25
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets lp \
    --detector yolo \
    --method white \
    --lp_threshold 0.25
```

---

### License Plate Anonymization (SAM3)

**Old command:**
```bash
python scripts/lp_anonymization_sam3.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --lp_anon white \
    --lp_threshold 0.5
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets lp \
    --method white \
    --lp_threshold 0.5
# Note: SAM3 is now the default detector!
```

---

### Body + License Plate (YOLO/YOLOX)

**Old command:**
```bash
python scripts/body_and_license_plate_anonymization.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --body_method pixel \
    --lp_method white
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets body lp \
    --detector yolo \
    --body_method pixel \
    --lp_method white
```

---

### Body + License Plate (SAM3)

**Old command:**
```bash
python scripts/body_lp_anonymization_sam3.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --body_anon gauss \
    --lp_anon white \
    --body_threshold 0.5 \
    --lp_threshold 0.5
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets body lp \
    --body_method gauss \
    --lp_method white \
    --body_threshold 0.5 \
    --lp_threshold 0.5
# Note: SAM3 is now the default detector!
```

---

### All Methods (Body)

**Old command:**
```bash
python scripts/body_anonymization.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --anon_function all
```

**New command:**
```bash
python scripts/anonymize.py \
    --image_dir /data/images \
    --output_dir /data/output \
    --targets body \
    --method all
```

---

## Key Differences

### 1. Output Naming Convention

**Old scripts:**
- Face: `{image}_anon_{method}.png`
- Body: `{image}_anon_{method}.png`
- LP: `{image}_anon_lp_{method}.png`
- Body+LP: `{image}_anon_body_{body_method}_lp_{lp_method}.png`

**New script:**
- Single target: `{image}_anon_{target}_{method}.png`
  - Example: `img001_anon_body_pixel.png`
- Multiple targets: `{image}_anon_{target1}+{target2}_{method}.png`
  - Example: `img001_anon_body+lp_pixel.png`
- All targets: `{image}_anon_face+body+lp_{method}.png`

### 2. Default Detector

**Old scripts:**
- `body_anonymization.py` → YOLO (yolo12l-person-seg-extended.pt)
- `license_plate_anonymization.py` → YOLOX (Autolane)
- `body_anonymization_sam3.py` → SAM3
- `lp_anonymization_sam3.py` → SAM3

**New script:**
- **Default: SAM3** for both body and license plates (based on evaluation showing 100% LP recall vs 35% for YOLOX)
- Use `--detector yolo` to use old detectors (YOLO for body, YOLOX for LP)

### 3. Statistics Output

**Old scripts:**
- Console output only
- Some scripts have progress bars

**New script:**
- Console output with detailed progress
- **`stats.json` file** with comprehensive statistics:
  - Total images, outputs, processing time
  - Throughput (images/sec)
  - Per-method counts
  - Detection statistics (bodies/LPs detected)

### 4. Resume Functionality

**Old scripts:**
- `body_anonymization.py` has `--resume` flag
- Others don't support resume

**New script:**
- All targets/methods support `--resume`
- Automatically skips images that have all outputs

---

## Legacy Scripts List

| Script | Purpose | Use New Script Instead |
|--------|---------|----------------------|
| `face_anonymization.py` | Face anonymization with RetinaFace masks | `anonymize.py --targets face` |
| `body_anonymization.py` | Body anonymization with YOLO (optimized) | `anonymize.py --targets body --detector yolo` |
| `license_plate_anonymization.py` | LP anonymization with YOLOX | `anonymize.py --targets lp --detector yolo` |
| `body_and_license_plate_anonymization.py` | Body+LP with YOLO/YOLOX | `anonymize.py --targets body lp --detector yolo` |
| `body_anonymization_sam3.py` | Body anonymization with SAM3 | `anonymize.py --targets body` |
| `lp_anonymization_sam3.py` | LP anonymization with SAM3 | `anonymize.py --targets lp` |
| `body_lp_anonymization_sam3.py` | Body+LP with SAM3 | `anonymize.py --targets body lp` |

---

## Backward Compatibility Notes

- Legacy scripts are **not deprecated** from a support perspective - they will continue to work
- However, **new features and improvements** will only be added to `anonymize.py`
- Legacy scripts are kept for:
  - Existing workflows/pipelines that depend on them
  - Compatibility with existing documentation
  - Users who prefer the old interface

---

## Getting Help

For help with the new unified script:
```bash
python scripts/anonymize.py --help
```

For questions or issues:
- Check the main README.md for examples
- See the repository documentation
- Open an issue on GitHub

---

**Last Updated:** March 17, 2026
