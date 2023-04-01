import argparse
import os
import torch
import pytorch_lightning as ptl
from pytorch_lightning.loggers import TensorBoardLogger

from detector.data import FontDataModule
from detector.model import *
from utils import get_current_tag


torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d",
    "--devices",
    nargs="*",
    type=int,
    default=[0],
    help="GPU devices to use (default: [0])",
)
parser.add_argument(
    "-b",
    "--single-batch-size",
    type=int,
    default=64,
    help="Batch size of single device (default: 64)",
)
parser.add_argument(
    "-c",
    "--checkpoint",
    type=str,
    default=None,
    help="Trainer checkpoint path (default: None)",
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="resnet18",
    choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    help="Model to use (default: resnet18)",
)
parser.add_argument(
    "-p",
    "--pretrained",
    action="store_true",
    help="Use pretrained model for ResNet (default: False)",
)
parser.add_argument(
    "-i",
    "--crop-roi-bbox",
    action="store_true",
    help="Crop ROI bounding box (default: False)",
)
parser.add_argument(
    "-a",
    "--augmentation",
    type=str,
    default=None,
    choices=["v1", "v2"],
    help="Augmentation strategy to use (default: None)",
)

args = parser.parse_args()

devices = args.devices
single_batch_size = args.single_batch_size

total_num_workers = os.cpu_count()
single_device_num_workers = total_num_workers // len(devices)


lr = 0.0001
b1 = 0.9
b2 = 0.999

lambda_font = 2.0
lambda_direction = 0.5
lambda_regression = 1.0

regression_use_tanh = False

num_warmup_epochs = 5
num_epochs = 100

log_every_n_steps = 100

num_device = len(devices)

data_module = FontDataModule(
    batch_size=single_batch_size,
    num_workers=single_device_num_workers,
    pin_memory=True,
    train_shuffle=True,
    val_shuffle=False,
    test_shuffle=False,
    regression_use_tanh=regression_use_tanh,
    train_transforms=args.augmentation,
    crop_roi_bbox=args.crop_roi_bbox,
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

if args.model == "resnet18":
    model = ResNet18Regressor(
        pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
    )
elif args.model == "resnet34":
    model = ResNet34Regressor(
        pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
    )
elif args.model == "resnet50":
    model = ResNet50Regressor(
        pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
    )
elif args.model == "resnet101":
    model = ResNet101Regressor(
        pretrained=args.pretrained, regression_use_tanh=regression_use_tanh
    )
else:
    raise NotImplementedError()

detector = FontDetector(
    model=model,
    lambda_font=lambda_font,
    lambda_direction=lambda_direction,
    lambda_regression=lambda_regression,
    lr=lr,
    betas=(b1, b2),
    num_warmup_iters=num_warmup_iter,
    num_iters=num_iters,
    num_epochs=num_epochs,
)

trainer.fit(detector, datamodule=data_module, ckpt_path=args.checkpoint)
trainer.test(detector, datamodule=data_module)
