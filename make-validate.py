#!/usr/bin/env python
from pathlib import Path
from random import shuffle


def validate_count(total, want):
    if want is None:
        want = 0.1

    if isinstance(want, float):
        want = int(total*want)

    want = min(total, want)
    return want

def main(scheme:str, number:[int, float] = 0.1):
    train_folder = Path(scheme).with_suffix('.train')
    val_folder = Path(scheme).with_suffix('.val')
    val_folder.mkdir(mode = 0o755, parents = False, exist_ok = True)

    classes = [cls for cls in train_folder.iterdir() if cls.is_dir()]
    for cls in train_folder.iterdir():
        if not cls.is_dir():
            continue
        samples = list(cls.glob('*.jpg'))+list(cls.glob('*.png'))
        shuffle(samples)
        nv = validate_count(len(samples), number)
        validates = samples[:nv]
        target = val_folder.joinpath(cls.name)
        target.mkdir(mode = 0o755, exist_ok=True)
        for sample in validates:
            sample.rename(target/sample.name)
        print(f'{cls} validate {len(validates)}' )


if __name__ == '__main__':
    from fire import Fire
    Fire(main)
