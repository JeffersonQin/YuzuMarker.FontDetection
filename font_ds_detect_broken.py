import sys
import traceback
import pickle
import os
import concurrent.futures
from tqdm import tqdm
import time
from font_dataset.font import load_fonts
import cv2

cjk_ratio = 3

train_cnt = 100
val_cnt = 5
test_cnt = 30

train_cnt_cjk = int(train_cnt * cjk_ratio)
val_cnt_cjk = int(val_cnt * cjk_ratio)
test_cnt_cjk = int(test_cnt * cjk_ratio)

dataset_path = "./dataset/font_img"
os.makedirs(dataset_path, exist_ok=True)

unqualified_log_file_name = f"unqualified_font_{time.time()}.txt"
runtime_exclusion_list = []

fonts, exclusion_rule = load_fonts()


def generate_dataset(dataset_type: str, cnt: int):
    dataset_bath_dir = os.path.join(dataset_path, dataset_type)
    os.makedirs(dataset_bath_dir, exist_ok=True)

    def _generate_single(args):
        i, j, font = args
        print(
            f"Checking {dataset_type} font: {font.path} {i} / {len(fonts)}, image {j}",
            end="\r",
        )

        if exclusion_rule(font):
            print(f"Excluded font: {font.path}")
            return
        if font.path in runtime_exclusion_list:
            print(f"Excluded font: {font.path}")
            return

        image_file_name = f"font_{i}_img_{j}.jpg"
        label_file_name = f"font_{i}_img_{j}.bin"

        image_file_path = os.path.join(dataset_bath_dir, image_file_name)
        label_file_path = os.path.join(dataset_bath_dir, label_file_name)

        # detect cache
        if (not os.path.exists(image_file_path)) or (
            not os.path.exists(label_file_path)
        ):
            print(
                f"Missing {dataset_type} font: {font.path} {i} / {len(fonts)}, image {j}"
            )

        # detect broken
        try:
            # check image
            cv2.imread(image_file_path)
            # check label
            with open(label_file_path, "rb") as f:
                pickle.load(f)
        except Exception as e:
            print(
                f"Broken {dataset_type} font: {font.path} {i} / {len(fonts)}, image {j}"
            )
            os.remove(image_file_path)
            os.remove(label_file_path)

        return

    work_list = []

    # divide len(fonts) into 64 parts and choose the third part for this script
    for i in range(len(fonts)):
        font = fonts[i]
        if font.language == "CJK":
            true_cnt = cnt * cjk_ratio
        else:
            true_cnt = cnt
        for j in range(true_cnt):
            work_list.append((i, j, font))

    for i in tqdm(range(len(work_list))):
        _generate_single(work_list[i])


generate_dataset("train", train_cnt)
generate_dataset("val", val_cnt)
generate_dataset("test", test_cnt)
