import sys
import traceback
import pickle
import os
import concurrent.futures
from tqdm import tqdm
import time
from font_dataset.font import load_fonts, DSFont
from font_dataset.layout import generate_font_image, TextSizeTooSmallException
from font_dataset.text import CorpusGeneratorManager, UnqualifiedFontException
from font_dataset.background import background_image_generator


global_script_index = int(sys.argv[1])
global_script_index_total = int(sys.argv[2])

print(f"Mission {global_script_index} / {global_script_index_total}")

num_workers = 32

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
corpus_manager = CorpusGeneratorManager()
images = background_image_generator()


def add_exclusion(font: DSFont, reason: str, dataset_base_dir: str, i: int, j: int):
    print(f"Excluded font: {font.path}, reason: {reason}")
    runtime_exclusion_list.append(font.path)
    with open(unqualified_log_file_name, "a+") as f:
        f.write(f"{font.path} # {reason}\n")
    for jj in range(j + 1):
        image_file_name = f"font_{i}_img_{jj}.jpg"
        label_file_name = f"font_{i}_img_{jj}.bin"

        image_file_path = os.path.join(dataset_base_dir, image_file_name)
        label_file_path = os.path.join(dataset_base_dir, label_file_name)

        if os.path.exists(image_file_path):
            os.remove(image_file_path)
        if os.path.exists(label_file_path):
            os.remove(label_file_path)


def generate_dataset(dataset_type: str, cnt: int):
    dataset_base_dir = os.path.join(dataset_path, dataset_type)
    os.makedirs(dataset_base_dir, exist_ok=True)

    def _generate_single(args):
        i, j, font = args
        print(
            f"Generating {dataset_type} font: {font.path} {i} / {len(fonts)}, image {j}"
        )

        if exclusion_rule(font):
            print(f"Excluded font: {font.path}")
            return
        if font.path in runtime_exclusion_list:
            print(f"Excluded font: {font.path}")
            return

        while True:
            try:
                image_file_name = f"font_{i}_img_{j}.jpg"
                label_file_name = f"font_{i}_img_{j}.bin"

                image_file_path = os.path.join(dataset_base_dir, image_file_name)
                label_file_path = os.path.join(dataset_base_dir, label_file_name)

                # detect cache
                if os.path.exists(image_file_path) and os.path.exists(label_file_path):
                    return

                im = next(images)
                im, label = generate_font_image(
                    im,
                    font,
                    corpus_manager,
                )

                im.save(image_file_path)
                pickle.dump(label, open(label_file_path, "wb"))
                return
            except UnqualifiedFontException as e:
                traceback.print_exc()
                add_exclusion(font, "unqualified font", dataset_base_dir, i, j)
                return
            except TextSizeTooSmallException as e:
                traceback.print_exc()
                continue
            except Exception as e:
                traceback.print_exc()
                add_exclusion(font, f"other: {repr(e)}", dataset_base_dir, i, j)
                return

    work_list = []

    # divide len(fonts) into 64 parts and choose the third part for this script
    for i in range(
        (global_script_index - 1) * len(fonts) // global_script_index_total,
        global_script_index * len(fonts) // global_script_index_total,
    ):
        font = fonts[i]
        if font.language == "CJK":
            true_cnt = cnt * cjk_ratio
        else:
            true_cnt = cnt
        for j in range(true_cnt):
            work_list.append((i, j, font))

    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     _ = list(
    #         tqdm(
    #             executor.map(_generate_single, work_list),
    #             total=len(work_list),
    #             leave=True,
    #             desc=dataset_type,
    #             miniters=1,
    #         )
    #     )

    for i in tqdm(range(len(work_list))):
        _generate_single(work_list[i])


generate_dataset("train", train_cnt)
generate_dataset("val", val_cnt)
generate_dataset("test", test_cnt)
