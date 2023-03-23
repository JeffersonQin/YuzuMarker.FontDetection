from font_dataset.fontlabel import FontLabel
from font_dataset.font import DSFont, load_font_with_exclusion
from . import config


import math
import os
import pickle
import torch
import torchvision.transforms as transforms
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from PIL import Image


class FontDataset(Dataset):
    def __init__(self, path: str, config_path: str = "configs/font.yml"):
        self.path = path
        self.fonts = load_font_with_exclusion(config_path)

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
            out[7:10] = 0.5
        out[10] = label.line_spacing / label.image_width
        out[11] = label.angle / 180.0 + 0.5

        return out

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load image
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
                transforms.ToTensor(),
            ]
        )
        image = transform(image)

        # Load label
        label_path = image_path.replace(".jpg", ".bin")
        with open(label_path, "rb") as f:
            label: FontLabel = pickle.load(f)

        # encode label
        label = self.fontlabel2tensor(label, label_path)

        return image, label


class FontDataModule(LightningDataModule):
    def __init__(
        self,
        config_path: str = "configs/font.yml",
        train_path: str = "./dataset/font_img/train",
        val_path: str = "./dataset/font_img/train",
        test_path: str = "./dataset/font_img/train",
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        test_shuffle: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataloader_args = kwargs
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.train_dataset = FontDataset(train_path, config_path)
        self.val_dataset = FontDataset(val_path, config_path)
        self.test_dataset = FontDataset(test_path, config_path)

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
