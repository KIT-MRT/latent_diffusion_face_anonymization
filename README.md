# Latent Diffusion Face Anonymisation LDFA

Please use the provided Docker container.

Prior to using this tool, please make sure that you have correctly set up the image, mask, anonymized, and weights volumes inside the `docker-compose.yml` file.
Furthermore, you can freely specify which GPU should be used.

`docker compose up`


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

```tex
@InProceedings{Klemp_2023_CVPR,
    author    = {Klemp, Marvin and R\"osch, Kevin and Wagner, Royden and Quehl, Jannik and Lauer, Martin},
    title     = {LDFA: Latent Diffusion Face Anonymization for Self-Driving Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3198-3204}
}
```
