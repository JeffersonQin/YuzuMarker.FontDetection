import argparse
import os
import torch
import pytorch_lightning as ptl
from pytorch_lightning.loggers import TensorBoardLogger

from detector.data import FontDataModule
from detector.model import FontDetector, ResNet18Regressor
from utils import get_current_tag


torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--devices', nargs='*', type=int, default=[0])

args = parser.parse_args()

devices = args.devices

final_batch_size = 128
single_device_num_workers = 24


lr = 0.00005
b1 = 0.9
b2 = 0.999

lambda_font = 4.0
lambda_direction = 0.5
lambda_regression = 1.0

regression_use_tanh = True

num_warmup_epochs = 1
num_epochs = 100

log_every_n_steps = 100

num_device = len(devices)

data_module = FontDataModule(
    batch_size=final_batch_size // num_device,
    num_workers=single_device_num_workers,
    pin_memory=True,
    train_shuffle=True,
    val_shuffle=False,
    test_shuffle=False,
    regression_use_tanh=regression_use_tanh,
)

num_iters = data_module.get_train_num_iter(num_device) * num_epochs
num_warmup_iter = data_module.get_train_num_iter(num_device) * num_warmup_epochs

model_name = f"{get_current_tag()}"

logger_unconditioned = TensorBoardLogger(
    save_dir=os.getcwd(), name="tensorboard", version=model_name
)

strategy = None if num_device == 1 else "ddp"

trainer = ptl.Trainer(
    max_epochs=num_epochs,
    logger=logger_unconditioned,
    devices=devices,
    accelerator="gpu",
    enable_checkpointing=True,
    log_every_n_steps=log_every_n_steps,
    strategy=strategy,
    deterministic=True,
)

model = ResNet18Regressor(regression_use_tanh=regression_use_tanh)

detector = FontDetector(
    model=model,
    lambda_font=lambda_font,
    lambda_direction=lambda_direction,
    lambda_regression=lambda_regression,
    lr=lr,
    betas=(b1, b2),
    num_warmup_iters=num_warmup_iter,
    num_iters=num_iters,
)

trainer.fit(detector, datamodule=data_module)
trainer.test(detector, datamodule=data_module)
