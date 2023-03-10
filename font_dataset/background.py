import os
from PIL import Image


__all__ = ["background_image_generator"]


def background_image_generator(path="./dataset/pixivimages"):
    image_list = os.listdir(path)
    image_list = [os.path.join(path, image) for image in image_list]

    while True:
        for image in image_list:
            yield image
