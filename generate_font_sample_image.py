import os
from PIL import Image, ImageDraw, ImageFont
from typing import List
from font_dataset.font import load_fonts, DSFont
from tqdm import tqdm

os.makedirs("./demo_fonts", exist_ok=True)

font_list, exclusion_rule = load_fonts()
font_list: List[DSFont] = list(filter(lambda x: not exclusion_rule(x), font_list))
font_list.sort(key=lambda x: x.path)

width = 320
height = 150
font_size = 32


def sample_text(font: DSFont) -> str:
    if font.language == "ja":
        return "こんにちは、世界\nフォント識別\nHello, world"
    if font.language == "ko":
        return "안녕하세요, 세계\n글꼴 인식하기\nHello, world"
    if font.language == "zh":
        return "你好，世界\n字体识别 字型辨識\nHello, world"
    if str(font.language).startswith("zh-Hans"):
        return "你好，世界\n字体识别\nHello, world"
    if str(font.language).startswith("zh-Hant"):
        return "你好，世界\n字型辨識\nHello, world"
    return "CJK字体\nCJKフォント\nCJK 글꼴"


for i, font in tqdm(enumerate(font_list)):
    img = Image.new("RGB", (width, height), (255, 255, 255))
    font_obj = ImageFont.truetype(font.path, font_size)
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(
        sample_text(font),
        font=font_obj,
        language=None if font.language == "CJK" else font.language,
    )
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), sample_text(font), font=font_obj, fill=(0, 0, 0))
    img.save(f"./demo_fonts/{i}.jpg")
