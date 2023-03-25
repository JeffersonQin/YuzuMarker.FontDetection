from font_dataset.fontlabel import FontLabel
from font_dataset.font import DSFont, load_font_with_exclusion
from . import config


import math
import os
import random
import pickle
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from PIL import Image


class RandomColorJitter(object):
    def __init__(
        self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05, preserve=0.2
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.preserve = preserve

    def __call__(self, batch):
        if random.random() < self.preserve:
            return batch

        image, label = batch
        text_color = label[2:5].clone().view(3, 1, 1)
        stroke_color = label[7:10].clone().view(3, 1, 1)

        brightness = random.uniform(1 - self.brightness, 1 + self.brightness)
        image = TF.adjust_brightness(image, brightness)
        text_color = TF.adjust_brightness(text_color, brightness)
        stroke_color = TF.adjust_brightness(stroke_color, brightness)

        contrast = random.uniform(1 - self.contrast, 1 + self.contrast)
        image = TF.adjust_contrast(image, contrast)
        text_color = TF.adjust_contrast(text_color, contrast)
        stroke_color = TF.adjust_contrast(stroke_color, contrast)

        saturation = random.uniform(1 - self.saturation, 1 + self.saturation)
        image = TF.adjust_saturation(image, saturation)
        text_color = TF.adjust_saturation(text_color, saturation)
        stroke_color = TF.adjust_saturation(stroke_color, saturation)

        hue = random.uniform(-self.hue, self.hue)
        image = TF.adjust_hue(image, hue)
        text_color = TF.adjust_hue(text_color, hue)
        stroke_color = TF.adjust_hue(stroke_color, hue)

        label[2:5] = text_color.view(3)
        label[7:10] = stroke_color.view(3)
        return image, label


class RandomCrop(object):
    def __init__(self, crop_factor: float = 0.1, preserve: float = 0.2):
        self.crop_factor = crop_factor
        self.preserve = preserve

    def __call__(self, batch):
        if random.random() < self.preserve:
            return batch

        image, label = batch
        width, height = image.size

        # use random value to decide scaling factor on x and y axis
        random_height = random.random() * self.crop_factor
        random_width = random.random() * self.crop_factor
        # use random value again to decide scaling factor for 4 borders
        random_top = random.random() * random_height
        random_left = random.random() * random_width
        # calculate new width and height and position
        top = int(random_top * height)
        left = int(random_left * width)
        height = int(height - random_height * height)
        width = int(width - random_width * width)
        # crop image
        image = TF.crop(image, top, left, height, width)

        label[[5, 6, 10]] = label[[5, 6, 10]] * (1 - random_height)
        return image, label


class FontDataset(Dataset):
    def __init__(
        self,
        path: str,
        config_path: str = "configs/font.yml",
        regression_use_tanh: bool = False,
        transforms: bool = False,
    ):
        self.path = path
        self.fonts = load_font_with_exclusion(config_path)
        self.regression_use_tanh = regression_use_tanh
        self.transforms = transforms

        self.images = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")
        ]
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def fontlabel2tensor(self, label: FontLabel, label_path) -> torch.Tensor:
        out = torch.zeros(12, dtype=torch.float)
        try:
            out[0] = self.fonts[label.font.path]
        except KeyError:
            print(f"Unqualified font: {label.font.path}")
            print(f"Label path: {label_path}")
            raise KeyError
        out[1] = 0 if label.text_direction == "ltr" else 1
        # [0, 1]
        out[2] = label.text_color[0] / 255.0
        out[3] = label.text_color[1] / 255.0
        out[4] = label.text_color[2] / 255.0
        out[5] = label.text_size / label.image_width
        out[6] = label.stroke_width / label.image_width
        if label.stroke_color:
            out[7] = label.stroke_color[0] / 255.0
            out[8] = label.stroke_color[1] / 255.0
            out[9] = label.stroke_color[2] / 255.0
        else:
            out[7:10] = out[2:5]
        out[10] = label.line_spacing / label.image_width
        out[11] = label.angle / 180.0 + 0.5

        return out

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")

        # Load label
        label_path = image_path.replace(".jpg", ".bin")
        with open(label_path, "rb") as f:
            label: FontLabel = pickle.load(f)

        # encode label
        label = self.fontlabel2tensor(label, label_path)

        # data augmentation
        if self.transforms:
            transform = transforms.Compose(
                [
                    RandomColorJitter(),
                    RandomCrop(),
                ]
            )
            image, label = transform((image, label))

        # resize and to tensor
        transform = transforms.Compose(
            [
                transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                transforms.ToTensor(),
            ]
        )
        image = transform(image)

        # normalize label
        if self.regression_use_tanh:
            label[2:12] = label[2:12] * 2 - 1

        return image, label


class FontDataModule(LightningDataModule):
    def __init__(
        self,
        config_path: str = "configs/font.yml",
        train_path: str = "./dataset/font_img/train",
        val_path: str = "./dataset/font_img/val",
        test_path: str = "./dataset/font_img/test",
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        test_shuffle: bool = False,
        train_transforms: bool = False,
        val_transforms: bool = False,
        test_transforms: bool = False,
        regression_use_tanh: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataloader_args = kwargs
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.train_dataset = FontDataset(
            train_path, config_path, regression_use_tanh, train_transforms
        )
        self.val_dataset = FontDataset(
            val_path, config_path, regression_use_tanh, val_transforms
        )
        self.test_dataset = FontDataset(
            test_path, config_path, regression_use_tanh, test_transforms
        )

    def get_train_num_iter(self, num_device: int) -> int:
        return math.ceil(
            len(self.train_dataset) / (self.dataloader_args["batch_size"] * num_device)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=self.train_shuffle,
            **self.dataloader_args,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=self.val_shuffle,
            **self.dataloader_args,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=self.test_shuffle,
            **self.dataloader_args,
        )
