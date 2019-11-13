#!/usr/bin/env python
# coding=utf-8
from pathlib import Path
from typing import AnyStr, Generator
from sys import stderr

from aip import AipImageCensor
from fire import Fire

APP_ID = '9722985'
API_KEY = 'rOnVUtKjftl9F4qvBKpvXOmw'
SECRET_KEY = 'kWM5YQSkKXBxmrmcMNfR2PokzrvP7rzK'

SCENES = [
    'ocr', 'public', 'politician', 'antiporn', 'webimage', 'disgust',
    'watermark', 'quality'
]

client = AipImageCensor(APP_ID, API_KEY, SECRET_KEY)


def iter_images(folder: str) -> Generator[Path, None, None]:
    for img in Path(folder).rglob('*.jpg'):
        yield img
    for img in Path(folder).rglob('*.png'):
        yield img


def file_content(fp: str) -> AnyStr:
    """ 读取图片 """
    with open(fp, 'rb') as f:
        return f.read()


def image_quality(filename: Path):
    result = client.imageCensorComb(file_content(filename), SCENES)
    result["_file_name"] = str(filename)
    print(result)


def dir_quality(dir: str = ".") -> None:
    for img in iter_images(dir):
        try:
            image_quality(img)
        except Exception as e:
            print(img, e, file=stderr)


if __name__ == '__main__':
    Fire(dir_quality)
