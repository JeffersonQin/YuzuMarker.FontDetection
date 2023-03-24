import torchmetrics
from . import config

from typing import Tuple, Dict, List, Any

import numpy as np
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as ptl


class ResNet18Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=False)
        self.model.fc = nn.Linear(512, config.FONT_COUNT + 12)

    def forward(self, X):
        X = self.model(X)
        # [0, 1]
        X[..., config.FONT_COUNT + 2 :] = X[..., config.FONT_COUNT + 2 :].sigmoid()
        return X


class FontDetectorLoss(nn.Module):
    def __init__(self, lambda_font, lambda_direction, lambda_regression):
        super().__init__()
        self.category_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.lambda_font = lambda_font
        self.lambda_direction = lambda_direction
        self.lambda_regression = lambda_regression

    def forward(self, y_hat, y):
        font_cat = self.category_loss(y_hat[..., : config.FONT_COUNT], y[..., 0].long())
        direction_cat = self.category_loss(
            y_hat[..., config.FONT_COUNT : config.FONT_COUNT + 2], y[..., 1].long()
        )
        regression = self.regression_loss(
            y_hat[..., config.FONT_COUNT + 2 :], y[..., 2:]
        )
        return (
            self.lambda_font * font_cat
            + self.lambda_direction * direction_cat
            + self.lambda_regression * regression
        )


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class FontDetector(ptl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lambda_font: float,
        lambda_direction: float,
        lambda_regression: float,
        lr: float,
        betas: Tuple[float, float],
        num_warmup_iters: int,
        num_iters: int,
    ):
        super().__init__()
        self.model = model
        self.loss = FontDetectorLoss(lambda_font, lambda_direction, lambda_regression)
        self.font_accur_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.FONT_COUNT
        )
        self.direction_accur_train = torchmetrics.Accuracy(
            task="multiclass", num_classes=2
        )
        self.font_accur_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=config.FONT_COUNT
        )
        self.direction_accur_val = torchmetrics.Accuracy(
            task="multiclass", num_classes=2
        )
        self.lr = lr
        self.betas = betas
        self.num_warmup_iters = num_warmup_iters
        self.num_iters = num_iters

    def forward(self, x):
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss, "pred": y_hat, "target": y}

    def training_step_end(self, outputs):
        y_hat = outputs["pred"]
        y = outputs["target"]
        self.log(
            "train_font_accur",
            self.font_accur_train(y_hat[..., : config.FONT_COUNT], y[..., 0]),
        )
        self.log(
            "train_direction_accur",
            self.direction_accur_train(
                y_hat[..., config.FONT_COUNT : config.FONT_COUNT + 2], y[..., 1]
            ),
        )

    def on_train_epoch_end(self) -> None:
        self.font_accur_train.reset()
        self.direction_accur_train.reset()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        X, y = batch
        y_hat = self.forward(X)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.font_accur_val.update(y_hat[..., : config.FONT_COUNT], y[..., 0])
        self.direction_accur_val.update(
            y_hat[..., config.FONT_COUNT : config.FONT_COUNT + 2], y[..., 1]
        )
        return {"loss": loss, "pred": y_hat, "target": y}

    def on_validation_epoch_end(self):
        self.log("val_font_accur", self.font_accur_val.compute())
        self.log("val_direction_accur", self.direction_accur_val.compute())
        self.font_accur_val.reset()
        self.direction_accur_val.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=self.betas
        )
        self.scheduler = CosineWarmupScheduler(
            optimizer, self.num_warmup_iters, self.num_iters
        )
        return optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx: int = 0,
        *args,
        **kwargs
    ):
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs
        )
        self.log("lr", self.scheduler.get_last_lr()[0])
        self.scheduler.step()
