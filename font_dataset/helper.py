from fontTools.ttLib import TTFont


__all__ = ["char_in_font"]


def char_in_font(unicode_char, font_path):
    try:
        font = TTFont(font_path, fontNumber=0)
        for cmap in font["cmap"].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
        return False
    except Exception as e:
        return False
