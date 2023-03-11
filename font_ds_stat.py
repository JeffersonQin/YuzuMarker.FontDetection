import sys
import traceback
import pickle
import os
import concurrent.futures
from tqdm import tqdm
from font_dataset.font import load_fonts
from font_dataset.layout import generate_font_image
from font_dataset.text import CorpusGeneratorManager
from font_dataset.background import background_image_generator


cjk_ratio = 3

train_cnt = 100
val_cnt = 5
test_cnt = 30

train_cnt_cjk = int(train_cnt * cjk_ratio)
val_cnt_cjk = int(val_cnt * cjk_ratio)
test_cnt_cjk = int(test_cnt * cjk_ratio)

dataset_path = "./dataset/font_img"
os.makedirs(dataset_path, exist_ok=True)

fonts, exclusion_rule = load_fonts()


cnt = 0

for font in fonts:
    if exclusion_rule(font):
        print(f"Excluded font: {font.path}")
        continue

    if font.language == "CJK":
        cnt += cjk_ratio
    else:
        cnt += 1


print("Total training images:", train_cnt * cnt)
print("Total validation images:", val_cnt * cnt)
print("Total testing images:", test_cnt * cnt)

if os.path.exists(os.path.join(dataset_path, "train")):
    num_file_train = len(os.listdir(os.path.join(dataset_path, "train")))
else:
    num_file_train = 0

if os.path.exists(os.path.join(dataset_path, "val")):
    num_file_val = len(os.listdir(os.path.join(dataset_path, "val")))
else:
    num_file_val = 0

if os.path.exists(os.path.join(dataset_path, "test")):
    num_file_test = len(os.listdir(os.path.join(dataset_path, "test")))
else:
    num_file_test = 0

print("Total files generated:", num_file_train + num_file_val + num_file_test)
print("Total files target:", (train_cnt + val_cnt + test_cnt) * cnt * 2)

print(
    f"{(num_file_train + num_file_val + num_file_test) / ((train_cnt + val_cnt + test_cnt) * cnt * 2) * 100:.2f}% completed"
)
