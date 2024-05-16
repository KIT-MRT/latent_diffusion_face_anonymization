<p align="center">
    <img src="https://raw.githubusercontent.com/KIT-MRT/latent_diffusion_face_anonymization/pages/docs/static/images/favicon.svg?sanitize=true"
        height="130">
</p>

<p align="center">
    <a> <img src="https://img.shields.io/badge/Project%20page-green?style=flat&link=https%3A%2F%2Fkit-mrt.github.io%2Flatent_diffusion_face_anonymization%2F"/></a>
    <a> <img src="https://img.shields.io/badge/Paper-CVPRW23-1c75b8?style=flat&link=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent%2FCVPR2023W%2FE2EAD%2Fhtml%2FKlemp_LDFA_Latent_Diffusion_Face_Anonymization_for_Self-Driving_Applications_CVPRW_2023_paper.html"/></a>
</p>






# Latent Diffusion Face Anonymisation LDFA
This repository contains the code for the paper LDFA: Latent Diffusion Face Anonymization for Self-driving Applications.

## Structure
### Dockerfile
The dockerfile is used to start container which runs the [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) web UI for stable diffusion. LDFA uses the API to conveniently use a stable diffusion model for the anonymization of human faces.

### Scripts
`detect_faces.py` - This script uses [RetinaFace](https://github.com/serengil/retinaface) to detect faces on a given dataset.  
`ldfa_face_anon.py` - This script implements the LDFA anonymization method.  
`simple_face_anon.py` - This script implements the naive anonymization methods cropping, gaussian noise and pixelaziation which are applied on detected faces. 

### Test
The tests are not meant to be used as a unit test, but to show a quick script usage of our tooling. The tests are run on some samples from the [cityscapes](https://www.cityscapes-dataset.com/) dataset.
## Usage
Please use the provided Docker container. Make sure that you have Docker Compose V2. See [Diff between V1 and V2](https://docs.docker.com/compose/migrate/#what-are-the-functional-differences-between-compose-v1-and-compose-v2)

Prior to using this tool, please make sure that you have correctly set up the image, mask, anonymized, and weights volumes inside the `docker-compose.yml` file. 
Furthermore, you can freely specify which GPU should be used.
You can start the needed docker instances with `docker compose up`.
The script will look for all images in the given root folder. The default extension is `png`. If you want to use other extension, you can provide a flag to the corresponding python scripts, e.g. `--image_extension=jpg`.

Once the docker container is running you can generate masks using:
```shell
docker compose exec anon python3 /tool/scripts/detect_faces.py --image_dir=/data/images --mask_dir=/data/masks
```

and anonymize the detected faces using:

```shell
docker compose exec anon python3 /tool/scripts/ldfa_face_anon.py --image_dir=/data/images --mask_dir=/data/masks --output_dir=/data/anonymized
```

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
