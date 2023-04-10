import yaml
import os
from typing import Dict
import pickle


from .utils import get_files


__all__ = ["load_fonts", "DSFont"]


class DSFont:
    def __init__(self, path, language):
        self.path = path
        self.language = language


def load_fonts(config_path="configs/font.yml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ds_config = config["dataset"]
    ds_path = ds_config["path"]

    font_list = []

    for spec in ds_config["specs"]:
        for spec_path in spec["path"]:
            spec_path = os.path.join(ds_path, spec_path)
            spec_files = get_files(spec_path)

            if spec.keys().__contains__("rule"):
                rule = eval(spec["rule"])
            else:
                rule = None

            for file in spec_files:
                if rule is not None and not rule(file):
                    print("skip: " + file)
                    continue
                font_list.append(DSFont(str(file).replace("\\", "/"), spec["language"]))

    font_list.sort(key=lambda x: x.path)

    exclusion_list = ds_config["exclusion"]
    exclusion_list = [os.path.join(ds_path, path) for path in exclusion_list]

    def exclusion_rule(font: DSFont):
        for exclusion in exclusion_list:
            if os.path.samefile(font.path, exclusion):
                return True
        return False

    return font_list, exclusion_rule


def load_font_with_exclusion(
    config_path="configs/font.yml", cache_path="font_list_cache.bin"
) -> Dict:
    if os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    font_list, exclusion_rule = load_fonts(config_path)
    font_list = list(filter(lambda x: not exclusion_rule(x), font_list))
    font_list.sort(key=lambda x: x.path)
    print("font count: " + str(len(font_list)))
    ret = {font_list[i].path: i for i in range(len(font_list))}
    with open(cache_path, "wb") as f:
        pickle.dump(ret, f)
    return ret
