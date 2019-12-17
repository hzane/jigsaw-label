#!/usr/bin/env python
from hashlib import sha1
from pathlib import Path
from tqdm import tqdm

def main(root:str):
    images = [img for img in Path(root).rglob('*.jpg') if len(img.stem)<=16]
    images.extend([img for img in Path(root).rglob('*.png') if len(img.stem)<=16])

    for img in tqdm(images):
        name = sha1(img.name.encode()).hexdigest() + img.suffix
        img.rename(img.parent/name)


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
