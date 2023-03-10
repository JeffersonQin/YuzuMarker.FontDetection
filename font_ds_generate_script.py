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


global_script_index = int(sys.argv[1])
global_script_index_total = int(sys.argv[2])

print(f"Mission {global_script_index} / {global_script_index_total}")

num_workers = 32

cjk_ratio = 3

train_cnt = 100
val_cnt = 10
test_cnt = 30

train_cnt_cjk = int(train_cnt * cjk_ratio)
val_cnt_cjk = int(val_cnt * cjk_ratio)
test_cnt_cjk = int(test_cnt * cjk_ratio)

dataset_path = "./dataset/font_img"
os.makedirs(dataset_path, exist_ok=True)


fonts = load_fonts()
corpus_manager = CorpusGeneratorManager()
images = background_image_generator()


def generate_dataset(dataset_type: str, cnt: int):
    dataset_bath_dir = os.path.join(dataset_path, dataset_type)
    os.makedirs(dataset_bath_dir, exist_ok=True)

    def _generate_single(args):
        while True:
            try:
                i, j, font = args

                image_file_name = f"font_{i}_img_{j}.jpg"
                label_file_name = f"font_{i}_img_{j}.bin"

                image_file_path = os.path.join(dataset_bath_dir, image_file_name)
                label_file_path = os.path.join(dataset_bath_dir, label_file_name)

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
            except Exception as e:
                traceback.print_exc()
                continue

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
