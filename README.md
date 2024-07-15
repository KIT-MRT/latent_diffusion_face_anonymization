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
### Anaconda
First setup the anaconda environment  
`conda create -n ldfa python=3.10`  
then install pytorch with the correct cuda version:  
`conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia` and xformers  
`pip install xformers==v0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118`.
After this you need to install all necessary dependencies and the module itself with  
`pip install -r requirements.txt && python setup.py install`

Once the docker container is running you can generate masks using:
```shell
python3 detect_faces.py --image_dir=/data/images --mask_dir=/data/masks
```

and anonymize the detected faces using:

```shell
python3 face_anonymization.py --image_dir=/data/images --mask_dir=/data/masks --output_dir=/data/anonymized --anon_function ldfa
```

You can also use the other anonymization functions implemented. See `python3 face_anonymization.py --help` for more functions.

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
