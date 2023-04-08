from font_dataset.fontlabel import FontLabel
from font_dataset.font import DSFont, load_font_with_exclusion
from . import config


import math
import os
import random
import pickle
import torch
import torchvision
import torchvision.transforms.functional as TF
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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

        label[[5, 6, 10]] = label[[5, 6, 10]] / (1 - random_width)
        return image, label


class RandomRotate(object):
    def __init__(self, max_angle: int = 15, preserve: float = 0.2):
        self.max_angle = max_angle
        self.preserve = preserve

    def __call__(self, batch):
        if random.random() < self.preserve:
            return batch

        image, label = batch

        angle = random.uniform(-self.max_angle, self.max_angle)
        image = TF.rotate(image, angle)
        label[11] = label[11] + angle / 180
        return image, label


class RandomNoise(object):
    def __init__(self, max_noise: float = 0.05, preserve: float = 0.1):
        self.max_noise = max_noise
        self.preserve = preserve

    def __call__(self, image):
        if random.random() < self.preserve:
            return image
        return torch.clamp(
            image + torch.randn_like(image) * random.random() * self.max_noise, 0, 1
        )


class RandomDownSample(object):
    def __init__(self, max_ratio: float = 2, preserve: float = 0.5):
        self.max_ratio = max_ratio
        self.preserve = preserve

    def __call__(self, image):
        if random.random() < self.preserve:
            return image
        ratio = random.uniform(1, self.max_ratio)
        return TF.resize(
            image, (int(image.size[1] / ratio), int(image.size[0] / ratio))
        )


class RandomCropPreserveAspectRatio(object):
    def __call__(self, batch):
        image, label = batch
        width, height = image.size

        if width == height:
            return batch

        if width > height:
            x = random.randint(0, width - height)
            image = TF.crop(image, 0, x, height, height)
            label[[5, 6, 10]] = label[[5, 6, 10]] / height * width
        else:
            y = random.randint(0, height - width)
            image = TF.crop(image, y, 0, width, width)
        return image, label


class RandomHorizontalFlip(object):
    def __init__(self, preserve: float = 0.5):
        self.preserve = preserve

    def __call__(self, batch):
        if random.random() < self.preserve:
            return batch

        image, label = batch
        image = TF.hflip(image)
        label[11] = 1 - label[11]

        return image, label


class FontDataset(Dataset):
    def __init__(
        self,
        path: str,
        config_path: str = "configs/font.yml",
        regression_use_tanh: bool = False,
        transforms: str = None,
        crop_roi_bbox: bool = False,
        preserve_aspect_ratio_by_random_crop: bool = False,
    ):
        """Font dataset

        Args:
            path (str): path to the dataset
            config_path (str, optional): path to font config file. Defaults to "configs/font.yml".
            regression_use_tanh (bool, optional): whether use tanh as regression normalization. Defaults to False.
            transforms (str, optional): choose from None, 'v1', 'v2', 'v3'. Defaults to None.
            crop_roi_bbox (bool, optional): whether to crop text roi bbox, must be true when transform='v2' or 'v3'. Defaults to False.
            preserve_aspect_ratio_by_random_crop (bool, optional): whether to preserve aspect ratio by random cropping maximum squares. Defaults to False.
        """
        self.path = path
        self.fonts = load_font_with_exclusion(config_path)
        self.regression_use_tanh = regression_use_tanh
        self.transforms = transforms
        self.crop_roi_bbox = crop_roi_bbox

        self.images = [
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")
        ]
        self.images.sort()

        if transforms == "v2" or transforms == "v3":
            assert (
                crop_roi_bbox
            ), "crop_roi_bbox must be true when transform='v2' or 'v3'"

        if transforms is None:
            label_image_transforms = []
            image_transforms = [
                torchvision.transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                torchvision.transforms.ToTensor(),
            ]
        elif transforms == "v1":
            label_image_transforms = [
                RandomColorJitter(preserve=0.2),
                RandomCrop(preserve=0.2),
            ]
            image_transforms = [
                torchvision.transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                torchvision.transforms.ToTensor(),
            ]
        elif transforms == "v2":
            label_image_transforms = [
                RandomColorJitter(preserve=0.2),
                RandomCrop(crop_factor=0.54, preserve=0),
                RandomRotate(preserve=0.2),
            ]
            image_transforms = [
                torchvision.transforms.GaussianBlur(
                    random.randint(1, 3) * 2 - 1, sigma=(0.1, 5.0)
                ),
                torchvision.transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                torchvision.transforms.ToTensor(),
                RandomNoise(max_noise=0.05, preserve=0.1),
            ]
        elif transforms == "v3":
            label_image_transforms = [
                RandomColorJitter(preserve=0.2),
                RandomCrop(crop_factor=0.54, preserve=0),
                RandomRotate(preserve=0.2),
                RandomHorizontalFlip(preserve=0.5),
            ]
            image_transforms = [
                torchvision.transforms.GaussianBlur(
                    random.randint(1, 3) * 2 - 1, sigma=(0.1, 5.0)
                ),
                RandomDownSample(max_ratio=2, preserve=0.5),
                torchvision.transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                torchvision.transforms.ToTensor(),
                RandomNoise(max_noise=0.05, preserve=0.1),
            ]
        else:
            raise ValueError(f"Unknown transform: {transforms}")

        if preserve_aspect_ratio_by_random_crop:
            label_image_transforms.append(RandomCropPreserveAspectRatio())

        if len(label_image_transforms) == 0:
            self.transform_label_image = None
        else:
            self.transform_label_image = torchvision.transforms.Compose(
                label_image_transforms
            )
        if len(image_transforms) == 0:
            self.transform_image = None
        else:
            self.transform_image = torchvision.transforms.Compose(image_transforms)

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

        # preparation
        if (self.transforms == "v1") or (self.transforms is None):
            if self.crop_roi_bbox:
                left, top, width, height = label.bbox
                image = TF.crop(image, top, left, height, width)
                label.image_width = width
                label.image_height = height
        elif self.transforms == "v2" or self.transforms == "v3":
            # crop from 30% to 130% of bbox
            left, top, width, height = label.bbox

            right = left + width
            bottom = top + height

            width_delta = width * 0.07
            height_delta = height * 0.07

            left = max(0, int(left - width_delta))
            top = max(0, int(top - height_delta))

            right = min(image.width, int(right + width_delta))
            bottom = min(image.height, int(bottom + height_delta))

            width = right - left
            height = bottom - top

            image = TF.crop(image, top, left, height, width)
            label.image_width = width
            label.image_height = height

        # encode label
        label = self.fontlabel2tensor(label, label_path)

        # transform
        if self.transform_label_image is not None:
            image, label = self.transform_label_image((image, label))
        if self.transform_image is not None:
            image = self.transform_image(image)

        # normalize label
        if self.regression_use_tanh:
            label[2:12] = label[2:12] * 2 - 1

        return image, label


class FontDataModule(LightningDataModule):
    def __init__(
        self,
        config_path: str = "configs/font.yml",
        train_paths: List[str] = ["./dataset/font_img/train"],
        val_paths: List[str] = ["./dataset/font_img/val"],
        test_paths: List[str] = ["./dataset/font_img/test"],
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        test_shuffle: bool = False,
        train_transforms: bool = None,
        val_transforms: bool = None,
        test_transforms: bool = None,
        crop_roi_bbox: bool = False,
        preserve_aspect_ratio_by_random_crop: bool = False,
        regression_use_tanh: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataloader_args = kwargs
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.train_dataset = ConcatDataset(
            [
                FontDataset(
                    train_path,
                    config_path,
                    regression_use_tanh,
                    train_transforms,
                    crop_roi_bbox,
                    preserve_aspect_ratio_by_random_crop,
                )
                for train_path in train_paths
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                FontDataset(
                    val_path,
                    config_path,
                    regression_use_tanh,
                    val_transforms,
                    crop_roi_bbox,
                    preserve_aspect_ratio_by_random_crop,
                )
                for val_path in val_paths
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                FontDataset(
                    test_path,
                    config_path,
                    regression_use_tanh,
                    test_transforms,
                    crop_roi_bbox,
                    preserve_aspect_ratio_by_random_crop,
                )
                for test_path in test_paths
            ]
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
