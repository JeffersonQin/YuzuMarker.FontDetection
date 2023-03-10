import yaml
import os

from .utils import get_files


__all__ = ["load_fonts", "DSFont"]


class DSFont:
    def __init__(self, path, language):
        self.path = path
        self.language = language


def load_fonts(config_path="configs/font.yml") -> list[DSFont]:
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
                font_list.append(DSFont(file, spec["language"]))

    font_list.sort(key=lambda x: x.path)
    return font_list
