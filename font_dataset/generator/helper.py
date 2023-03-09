from fontTools.ttLib import TTFont


__all__ = ["char_in_font"]


def char_in_font(unicode_char, font_path):
    font = TTFont(font_path)
    for cmap in font["cmap"].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False
