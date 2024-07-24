import numpy as np
from pathlib import Path
from PIL import Image


class Face:
    def __init__(self, face_bounding_box_list: list[int]):
        self.bounding_box = BoundingBox(face_bounding_box_list)
        self.face_cutout: Image.Image
        self.face_cutout_resized: Image.Image
        self.mask_image: Image.Image
        self.mask_image_resized: Image.Image
        self.face_anon: Image.Image

    def set_face_cutout(self, image: np.ndarray):
        self.face_cutout = Image.fromarray(image[self.bounding_box.get_slice_area()])

    def set_mask_image(self, mask_image: np.ndarray):
        self.mask_image = Image.fromarray(mask_image)

    def resize(self, width: int, height: int):
        self.face_cutout_resized = self.face_cutout.resize((width, height))
        self.mask_image_resized = self.mask_image.resize((width, height))

    def add_anon_face_to_image(self, image: np.ndarray):
        image[self.bounding_box.get_slice_area()] = np.array(self.face_cutout)
        return image

    def save(self, save_path: Path, img_id: int, face_id: int):
        self.face_anon.save(f"{save_path}/face_anon_{img_id}_{face_id}.png")
        self.mask_image.save(f"{save_path}/mask_{img_id}_{face_id}.png")


class BoundingBox:
    def __init__(self, bounding_box_list: list[int]):
        self.xtl = bounding_box_list[0]
        self.ytl = bounding_box_list[3]
        self.xbr = bounding_box_list[2]
        self.ybr = bounding_box_list[1]
        self.confidence = bounding_box_list[4]

    def get_slice_area(self):
        return (slice(self.ytl, self.ybr), slice(self.xtl, self.xbr))


def add_face_cutout_and_mask_img(faces: list[Face], image: np.ndarray):
    for face in faces:
        face.set_face_cutout(image)
        mask_image_np = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        mask_image_np[face.bounding_box.get_slice_area()] = 255
        face.set_mask_image(mask_image_np)
    return faces
