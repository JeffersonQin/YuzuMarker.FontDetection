import os

from font_dataset.utils import get_files

directory_path = "dataset"

files = get_files(directory_path)


def clean_name(file_name: str):
    return (
        file_name.replace("&", "-")
        .replace("@", "-")
        .replace("#", "-")
        .replace("$", "-")
    )


for file_name in files:
    new_file_name = clean_name(file_name)
    if new_file_name != file_name:
        os.rename(file_name, new_file_name)
        print(f"Renamed {file_name} to {new_file_name}")
