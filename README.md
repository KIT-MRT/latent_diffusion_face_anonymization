<p align="center">
    <img src="https://raw.githubusercontent.com/KIT-MRT/latent_diffusion_face_anonymization/pages/docs/static/images/favicon.svg?sanitize=true"
        height="130">
</p>

<p align="center">
    <a href="https://kit-mrt.github.io/latent_diffusion_face_anonymization/"> <img src="https://img.shields.io/badge/Project%20page-green?style=flat"/></a>
    <a href="https://openaccess.thecvf.com/content/CVPR2023W/E2EAD/papers/Klemp_LDFA_Latent_Diffusion_Face_Anonymization_for_Self-Driving_Applications_CVPRW_2023_paper.pdf"> <img src="https://img.shields.io/badge/Paper-CVPRW23-1c75b8?style=flat"/></a>
</p>






# Latent Diffusion Face Anonymisation LDFA
This repository contains the code for the paper LDFA: Latent Diffusion Face Anonymization for Self-driving Applications.

## Quick Start (Recommended)

Use the new unified `anonymize.py` script for all anonymization tasks:

```bash
# Anonymize bodies and license plates with SAM3 (default, high accuracy)
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets body lp --method pixel

# Anonymize everything (face, body, license plates)
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets face body lp --mask_dir /data/masks --method lda

# Use all anonymization methods
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets body --method all

# Use old YOLO/YOLOX detectors instead of SAM3
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets body lp --detector yolo --method pixel
```

For more options: `python scripts/anonymize.py --help`

## Structure
### Dockerfile
The dockerfile is used to start container which runs the [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) web UI for stable diffusion. LDFA uses the API to conveniently use a stable diffusion model for the anonymization of human faces.

### Scripts
**Main Script (Recommended):**
- `anonymize.py` - Unified anonymization for faces, bodies, and license plates with SAM3 or YOLO detectors

**Legacy Scripts** (in `scripts/legacy/`):
- `detect_faces.py` - Uses [RetinaFace](https://github.com/serengil/retinaface) to detect faces
- `face_anonymization.py` - Face anonymization (use `anonymize.py --targets face` instead)
- `body_anonymization.py` - Body anonymization with YOLO (use `anonymize.py --targets body --detector yolo` instead)
- `license_plate_anonymization.py` - LP anonymization with YOLOX (use `anonymize.py --targets lp --detector yolo` instead)

See `scripts/legacy/README.md` for migration guide.

### Test
The tests are not meant to be used as a unit test, but to show a quick script usage of our tooling. The tests are run on some samples from the [cityscapes](https://www.cityscapes-dataset.com/) dataset.
## Usage
### Anaconda
First setup the anaconda environment  
`conda create -n ldfa python=3.10`  
then install pytorch with the correct cuda version:  
`conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia` and xformers  
`pip install xformers==v0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118`.
After this you need to install all necessary dependencies and the module itself with  
`pip install -r requirements.txt && python setup.py install`

The stable diffusion interface is not included in the anaconda environment. You can use the docker container to run the stable diffusion interface.
### Docker 
First build the docker image with  
```shell
docker build -t ldfa .
```
Then you can run the docker container with  
```shell
docker run -p 7860:7860 ldfa 
```

### Unified Anonymization (Recommended)

The new `anonymize.py` script provides a unified interface for all anonymization tasks with SAM3 or YOLO detectors:

```shell
# Body and license plate anonymization with SAM3 (default, high accuracy)
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets body lp --method pixel

# Face anonymization with LDA (requires docker container running)
# First, generate masks:
python scripts/detect_faces.py --image_dir /data/images --mask_dir /data/masks

# Then anonymize:
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets face --mask_dir /data/masks --method lda

# Different methods per target
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets body lp --body_method gauss --lp_method white

# Process with all methods (white, gauss, pixel, lda)
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets body --method all

# Resume interrupted processing
python scripts/anonymize.py --image_dir /data/images --output_dir /data/output \
    --targets body lp --method pixel --resume
```

**Available Detectors:**
- **SAM3** (default): Segment Anything 3 - higher accuracy (100% LP recall vs 35% for YOLOX)
- **YOLO/YOLOX**: Original detectors - use with `--detector yolo`

**Available Methods:**
- `white` - White fill
- `gauss` - Gaussian blur
- `pixel` - Pixelization
- `lda` - Latent Diffusion Anonymization (requires stable diffusion API)

For more options: `python scripts/anonymize.py --help`

### Legacy Scripts

For backward compatibility, legacy scripts are available in `scripts/legacy/`:

**Face Anonymization (Legacy):**
```shell
python scripts/legacy/face_anonymization.py --image_dir=/data/images --mask_dir=/data/masks \
    --output_dir=/data/anonymized --anon_function lda
```

**Body Anonymization (Legacy):**
```shell
python scripts/legacy/body_anonymization.py --image_dir=/data/images \
    --output_dir=/data/anonymized --anon_function lda
```

See `scripts/legacy/README.md` for migration guide and more examples.

# Citation

If you are using LDFA in your research, please consider to cite us.

```bibtex
@InProceedings{Klemp_2023_CVPR,
    author    = {Klemp, Marvin and R\"osch, Kevin and Wagner, Royden and Quehl, Jannik and Lauer, Martin},
    title     = {LDFA: Latent Diffusion Face Anonymization for Self-Driving Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3198-3204}
}
```
