from typing import Tuple

__all__ = ["generate_font_image", "TextSizeTooSmallException"]


epislon = 1e-6
render_calculation_size = 128

# text direction
ltr_ratio = 0.5
ttb_ratio = 0.5

assert ltr_ratio + ttb_ratio - 1 < epislon

# text length
short_ratio = 0.1
median_ratio = 0.6
long_ratio = 0.3

from .text import CorpusGenerationConfig, CorpusGeneratorManager

short_condition = CorpusGenerationConfig(
    min_num_line=1, max_num_line=1, min_num_char_per_line=2, max_num_char_per_line=5
)

median_condition = CorpusGenerationConfig(
    min_num_line=1, max_num_line=4, min_num_char_per_line=1, max_num_char_per_line=20
)

long_condition = CorpusGenerationConfig(
    min_num_line=5, max_num_line=10, min_num_char_per_line=1, max_num_char_per_line=20
)

assert short_ratio + median_ratio + long_ratio - 1 < epislon

# text color
gray_ratio = 0.3
color_ratio = 0.7

# whether use stroke, only stroke when color
pure_color_ratio = 0.5
stroke_color_ratio = 0.5

assert pure_color_ratio + stroke_color_ratio - 1 < epislon

# stroke width
stroke_width_max_ratio = 0.25

assert gray_ratio + color_ratio - 1 < epislon

# clip size ratio
clip_width_max_ratio = 0.8
clip_width_min_ratio = 0.3
clip_width_height_min_ratio = 0.75
clip_width_height_max_ratio = 1.25

# text longer edge ratio
text_longer_max_ratio = 1.0
text_longer_min_ratio = 0.6

# line spacing
line_spacing_max_ratio = 1.5
line_spacing_min_ratio = 0.0

# rotation
no_rotation_ratio = 0.3
rotation_ratio = 0.7

assert no_rotation_ratio + rotation_ratio - 1 < epislon

# in degree
rotation_max_angle = 30

text_size_min = 15

# ratio of dataset size for cjk
cjk_distribution = {
    "ja": 0.3,
    "ko": 0.2,
    "zh-Hans": 0.3,
    "zh-Hant": 0.07,
    "zh-Hant-HK": 0.06,
    "zh-Hant-TW": 0.06,
}

assert sum(cjk_distribution.values()) - 1 < epislon


import math
import random
import traceback
from PIL import Image, ImageDraw, ImageFont
from .fontlabel import FontLabel
from .font import DSFont


def random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def render_bbox(
    draw,
    xy,
    text: str,
    font=None,
    anchor=None,
    spacing=4,
    align="left",
    direction=None,
    features=None,
    language=None,
    stroke_width=0,
    embedded_color=False,
):
    if ("\n" in text or "\r" in text) and direction == "ttb":
        lines = text.splitlines(keepends=False)
        height = 0
        width = 0
        x = 0
        y = 0
        for i, line in enumerate(lines):
            bbox = draw.textbbox(
                (0, 0),
                line,
                font,
                anchor,
                spacing,
                align,
                direction,
                features,
                language,
                stroke_width,
                embedded_color,
            )
            height = max(height, bbox[3] - bbox[1])
            width += bbox[2] - bbox[0]
            if i > 0:
                width += spacing
            else:
                x = bbox[0]
                y = bbox[1]
        return x, y, x + width, y + height
    else:
        return draw.textbbox(
            xy,
            text,
            font,
            anchor,
            spacing,
            align,
            direction,
            features,
            language,
            stroke_width,
            embedded_color,
        )


def render_text(
    draw,
    xy,
    text,
    fill=None,
    font=None,
    anchor=None,
    spacing=4,
    align="left",
    direction=None,
    features=None,
    language=None,
    stroke_width=0,
    stroke_fill=None,
    embedded_color=False,
    *args,
    **kwargs,
):
    if ("\n" in text or "\r" in text) and direction == "ttb":
        lines = text.splitlines(keepends=False)
        margin_x = 0
        x, y = xy
        for i, line in enumerate(lines):
            bbox = draw.textbbox(
                (0, 0),
                line,
                font,
                anchor,
                spacing,
                align,
                direction,
                features,
                language,
                stroke_width,
                embedded_color,
            )
            draw.text(
                (x + margin_x, y),
                line,
                fill,
                font,
                anchor,
                spacing,
                align,
                direction,
                features,
                language,
                stroke_width,
                stroke_fill,
                embedded_color,
                *args,
                **kwargs,
            )
            margin_x += bbox[2] - bbox[0]
            margin_x += spacing
    else:
        draw.text(
            xy,
            text,
            fill,
            font,
            anchor,
            spacing,
            align,
            direction,
            features,
            language,
            stroke_width,
            stroke_fill,
            embedded_color,
            *args,
            **kwargs,
        )


def RGB2RGBA(color):
    if color is None:
        return None
    return color + (255,)


class TextSizeTooSmallException(Exception):
    def __init__(self):
        super().__init__(f"Text Size Too Small")


def generate_font_image(
    img_path: str, font: DSFont, corpus_manager: CorpusGeneratorManager
) -> Tuple[Image.Image, FontLabel]:
    im = Image.open(img_path)
    # crop image
    width, height = im.size
    clip_width = random.randint(
        int(width * clip_width_min_ratio), int(width * clip_width_max_ratio)
    )
    clip_height = random.randint(
        int(clip_width * clip_width_height_min_ratio),
        int(clip_width * clip_width_height_max_ratio),
    )
    if clip_height > height:
        clip_height = height
    clip_x = random.randint(0, width - clip_width)
    clip_y = random.randint(0, height - clip_height)
    im = im.crop((clip_x, clip_y, clip_x + clip_width, clip_y + clip_height))

    # language
    render_language = font.language
    if render_language == "CJK":
        render_language = random.choices(
            list(cjk_distribution.keys()), list(cjk_distribution.values())
        )[0]
    elif render_language == "zh":
        render_language = random.choice(["zh-Hans", "zh-Hant"])

    # text direction
    if random.random() < ltr_ratio:
        text_direction = "ltr"
    else:
        text_direction = "ttb"

    # text length
    if random.random() < short_ratio:
        text = corpus_manager.generate(short_condition, font, render_language)
    elif random.random() < median_ratio:
        text = corpus_manager.generate(median_condition, font, render_language)
    else:
        text = corpus_manager.generate(long_condition, font, render_language)

    # text color & stroke
    if random.random() < gray_ratio:
        text_color = random.randint(0, 255)
        text_color = (text_color, text_color, text_color)
        # no stroke in gray
        stroke_ratio = 0
        stroke_color = None
        im = im.convert("L")
    else:
        text_color = random_color()
        # whether use stroke
        if random.random() < pure_color_ratio:
            stroke_ratio = 0
            stroke_color = None
        else:
            stroke_ratio = random.random() * stroke_width_max_ratio
            stroke_color = random_color()

    # line spacing
    line_spacing_ratio = (
        random.random() * (line_spacing_max_ratio - line_spacing_min_ratio)
        + line_spacing_min_ratio
    )

    # calculate render ratio
    render_calculation_stroke_width = int(stroke_ratio * render_calculation_size)
    render_calculation_line_spacing = int(line_spacing_ratio * render_calculation_size)

    pil_font = ImageFont.truetype(font.path, size=render_calculation_size)
    text_bbox = render_bbox(
        ImageDraw.Draw(im),
        (0, 0),
        text,
        font=pil_font,
        direction=text_direction,
        spacing=render_calculation_line_spacing,
        stroke_width=render_calculation_stroke_width,
        language=render_language,
    )
    (
        render_calculation_width_no_rotation,
        render_calculation_height_no_rotation,
    ) = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
    render_calculation_font_x_no_rotation = text_bbox[0]
    render_calculation_font_y_no_rotation = text_bbox[1]

    if random.random() < no_rotation_ratio:
        render_angle = 0

        render_calculation_width = render_calculation_width_no_rotation
        render_calculation_height = render_calculation_height_no_rotation
    else:
        render_angle = random.randint(-rotation_max_angle, rotation_max_angle)

        render_calculation_width = int(
            render_calculation_width_no_rotation
            * math.cos(math.radians(abs(render_angle)))
            + render_calculation_height_no_rotation
            * math.sin(math.radians(abs(render_angle)))
        )
        render_calculation_height = int(
            render_calculation_width_no_rotation
            * math.sin(math.radians(abs(render_angle)))
            + render_calculation_height_no_rotation
            * math.cos(math.radians(abs(render_angle)))
        )

    # calculate render size
    render_ratio = (
        random.random() * (text_longer_max_ratio - text_longer_min_ratio)
        + text_longer_min_ratio
    )
    if render_calculation_width / render_calculation_height < clip_width / clip_height:
        # height is the limit
        render_height = int(clip_height * render_ratio)
        render_width = int(
            render_calculation_width / render_calculation_height * render_height
        )
    else:
        # width is the limit
        render_width = int(clip_width * render_ratio)
        render_height = int(
            render_calculation_height / render_calculation_width * render_width
        )

    # calculate text size
    text_size = int(render_calculation_size * render_height / render_calculation_height)

    if text_size < text_size_min:
        raise TextSizeTooSmallException()

    render_width_no_rotation = int(
        render_calculation_width_no_rotation / render_calculation_height * render_height
    )
    render_height_no_rotation = int(
        render_calculation_height_no_rotation
        / render_calculation_height
        * render_height
    )
    render_font_x_no_rotation = int(
        render_calculation_font_x_no_rotation
        / render_calculation_height
        * render_height
    )
    render_font_y_no_rotation = int(
        render_calculation_font_y_no_rotation
        / render_calculation_height
        * render_height
    )
    stroke_width = int(text_size * stroke_ratio)
    line_spacing = int(text_size * line_spacing_ratio)

    # calculate render position
    render_x = random.randint(0, clip_width - render_width)
    render_y = random.randint(0, clip_height - render_height)

    font_image = Image.new(
        "RGBA",
        (render_width_no_rotation, render_height_no_rotation),
        (0, 0, 0, 0),
    )
    pil_font = ImageFont.truetype(font.path, size=text_size)
    render_text(
        ImageDraw.Draw(font_image),
        (-render_font_x_no_rotation, -render_font_y_no_rotation),
        text,
        font=pil_font,
        fill=RGB2RGBA(text_color),
        direction=text_direction,
        spacing=line_spacing,
        stroke_width=stroke_width,
        stroke_fill=RGB2RGBA(stroke_color),
        language=render_language,
    )
    if rotation_max_angle != 0:
        font_image = font_image.rotate(
            render_angle, expand=True, fillcolor=(0, 0, 0, 0)
        )

    im.paste(font_image, (render_x, render_y), font_image)
    return im, FontLabel(
        clip_width,
        clip_height,
        text,
        font,
        text_color,
        text_size,
        text_direction,
        stroke_width,
        stroke_color,
        line_spacing,
        render_language,
        (render_x, render_y, render_width, render_height),
        render_angle,
    )
