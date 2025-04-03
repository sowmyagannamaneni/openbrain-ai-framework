import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import cv2
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
# ---------------------- Utility Functions ---------------------- #
HU_min, HU_max = -200, 250
data_mean = 50.21997497685108
data_std = 68.47153712416372

def read_image(path):
    with open(path, 'rb') as file:
        img = pickle.load(file)
        return img

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def convert_to_PIL(img: np.array) -> PIL.Image:
    '''Convert normalized image array to PIL Image'''
    img = np.clip(img, 0, 1)
    return PIL.Image.fromarray((img * 255).astype(np.uint8))

def convert_to_np(img: PIL.Image) -> np.array:
    '''Convert PIL Image to numpy array'''
    return np.array(img).astype(np.float32) / 255

# ---------------------- Data Augmentation Class ---------------------- #
class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res
        seed = 42
        self.rng = np.random.default_rng(seed)
        self.p = 0.5
        self.n = 2

    def create_ops(self):
        ops = [
            (shear_x, self.shear),
            (shear_y, self.shear),
            (scale, self.scale),
            (translate_x, self.translate),
            (translate_y, self.translate),
            (posterize, self.posterize),
            (contrast, self.contrast),
            (brightness, self.brightness),
            (sharpness, self.sharpness),
            (identity, (0, 1, 1)),
        ]

        self.ops = [op for op in ops if op[1][2] != 0]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)

        # Low-res label resizing
        low_res_label = zoom(label, (self.low_res[0] / label.shape[0], self.low_res[1] / label.shape[1], 1.0), order=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))

        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        low_res_label = low_res_label.permute(2, 0, 1)

        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


# ---------------------- Dataset Reader ---------------------- #
class dataset_reader(Dataset):
    def __init__(self, base_dir, split, num_classes, transform=None):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir

        if split == "train":
            df = pd.read_csv(base_dir + '/training.csv')
            self.sample_list = [base_dir + '/' + sample_pth.split('/' + base_dir.split('/')[-1] + '/')[-1] for sample_pth in df["image_pth"]]
            self.masks_list = [base_dir + '/' + sample_pth.split('/' + base_dir.split('/')[-1] + '/')[-1] for sample_pth in df["mask_pth"]]
            self.num_classes = num_classes

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            data = read_image(self.sample_list[idx])
            data = np.clip(data, HU_min, HU_max)
            data = (data - HU_min) / (HU_max - HU_min) * 255.0

            data = np.float32(data)
            data = (data - data_mean) / data_std
            data = (data - data.min()) / (data.max() - data.min() + 0.00000001)
            h, w, d = data.shape

            data = np.float32(data)

            mask = read_image(self.masks_list[idx])
            mask = np.float32(mask)

            if self.num_classes == 12:
                mask[mask == 13] = 12

            image = np.float32(data)
            label = np.float32(mask)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
