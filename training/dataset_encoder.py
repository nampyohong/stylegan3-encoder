import os

import PIL.Image
import torch
from torchvision.transforms import (Compose, Resize, RandomHorizontalFlip, 
                                    ToTensor, Normalize)

# data utils
"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir_):
    images = []
    assert os.path.isdir(dir_), '%s is not a valid directory' % dir_
    for root, _, fnames in sorted(os.walk(dir_)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


# dataset
class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, mode='train'):
        assert mode in ['train', 'test', 'inference']
        self.paths = sorted(make_dataset(dataset_dir))
        transforms_dict = {
            'train': Compose([
                Resize((256, 256)),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'test': Compose([
                Resize((256, 256)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'inference': Compose([
                Resize((256, 256)),
                ToTensor(),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        self.transforms = transforms_dict.get(mode)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        pil_img = PIL.Image.open(self.paths[i]).convert('RGB')
        return self.transforms(pil_img)


# TODO : implement distributed sampler for ddp
