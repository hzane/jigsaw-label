#!/usr/bin/env python

from PIL import Image
from pathlib import Path
from fire import Fire
from typing import Generator


def all_images(root: str) -> Generator[Path, None, None]:
    for p in Path(root).rglob("*.jpg"):
        yield p
    for p in Path(root).rglob("*.png"):
        yield p


def clean(root: str = ".",
          recycle: str = "../recycle",
          ratio: int = 2,
          min_width: int = 400,
          min_height: int = 400) -> None:
    dest = Path(recycle)
    dest.mkdir(0o755, parents=False, exist_ok=True)
    for imgfn in all_images(root):
        try:
            img = Image.open(imgfn)
        except IOError:
            imgfn.rename(dest.joinpath(imgfn.name))
        else:
            width, height = img.size
            if width > height * ratio or height > width * ratio:
                print(imgfn, width, height)
                imgfn.rename(dest.joinpath(imgfn.name))
            elif width < min_width and height < min_height:
                print(imgfn, width, height, "dropped")
                imgfn.unlink()

            img.close()


if __name__ == "__main__":
    Fire(clean)
