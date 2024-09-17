import unittest
import numpy as np
import logging
import os
import torch
from pathlib import Path

import diffusion_face_anonymisation.utils as dfa_utils
import diffusion_face_anonymisation.io_functions as dfa_io
from diffusion_face_anonymisation.face import add_face_cutout_and_mask_img

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from PIL import Image


class SDPipelineWrapper:
    INFERENCE_STEPS = 50

    def __init__(self, prompt: str, gpu_id=0):
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
        ).to(torch.device(f"cuda:{gpu_id}"))
        self.prompt = prompt

    def run(self, image: Image.Image, mask: Image.Image):
        return self.pipe(
            prompt=self.prompt,
            image=image,
            mask_image=mask,
            strength=0.75,
            num_inference_steps=self.INFERENCE_STEPS,
            generator=torch.Generator("cuda").manual_seed(2353552617),
        )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_ldfa.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class FaceInpaintingTest(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_image_base_path = os.path.abspath(os.path.join(file_dir, "leftImg8Bit"))
    test_mask_base_path = os.path.abspath(os.path.join(file_dir, "annotations"))
    sd_pipe = SDPipelineWrapper("", gpu_id=0)

    def test_(self):
        json_files = dfa_io.glob_files_by_extension(self.test_mask_base_path, "json")
        png_files = dfa_io.glob_files_by_extension(self.test_image_base_path, "png")
        image_mask_dict = {}
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(
            json_files, image_mask_dict, "mask_file"
        )
        image_mask_dict = dfa_utils.add_file_paths_to_image_mask_dict(
            png_files, image_mask_dict, "image_file"
        )

        for img_id, entry in enumerate(image_mask_dict.values()):
            image = dfa_utils.preprocess_image(entry["image_file"])
            faces = dfa_io.get_faces_from_file(entry["mask_file"])
            faces = add_face_cutout_and_mask_img(faces, image)
            ldfa_image = np.array(image)
            for j, face in enumerate(faces):
                face.resize(width=512, height=512)

                face.mask_image_resized = self.sd_pipe.pipe.mask_processor.blur(
                    face.mask_image_resized, blur_factor=3
                )
                anon_image = self.sd_pipe.run(
                    face.face_cutout_resized, face.mask_image_resized
                )
                face.face_anon = anon_image.images[0]
                ldfa_image = face.add_anon_face_to_image(ldfa_image)
                face.save(Path("/storage_local/roesch/ldba/faces_tmp"), img_id, j)
            Image.fromarray(ldfa_image).save(
                f"/storage_local/roesch/ldba/faces_tmp/anon_{img_id}.png"
            )


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
