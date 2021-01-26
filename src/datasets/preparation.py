import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class DatasetSimple(Dataset):
    """
      Args:
        root (string): Root directory path.
        frame_list_path (string): Frame list path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        sample_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    """

    def __init__(self, root, frame_list_path, transform=None, sample_transform=None):
        self.root = root
        self.frame_list = np.loadtxt(frame_list_path, skiprows=1, dtype=str)
        self.transform = transform
        self.sample_transform = sample_transform

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        image_name = self.frame_list[index]
        image_path = os.path.join(self.root, image_name)

        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, image_name


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list | ndarray
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros(2, dtype=np.float32)

    top_left_corner = box[0:2]
    box_width = box[2]
    box_height = box[3]
    center[0] = top_left_corner[0] + box_width * 0.5
    center[1] = top_left_corner[1] + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


class DatasetDetections(DatasetSimple):
    def __getitem__(self, index):
        frame_info = self.frame_list[index].split(",")
        image_name = frame_info[0]
        image_path = os.path.join(self.root, image_name)

        box = np.array(frame_info[1:], dtype=np.float32)
        center, scale = box_to_center_scale(box, 288, 384)

        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")

        if self.sample_transform:
            img = self.sample_transform(img, center, scale)

        if self.transform:
            img = self.transform(img)

        return img, image_name, (center, scale)
