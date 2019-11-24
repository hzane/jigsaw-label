#!/usr/bin/env python
# %%
from pathlib import Path
from typing import Generator, Union

import pandas as pd
import torch
import torch.nn as nn
from fire import Fire
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# batch_size = 80
# num_workers = 2


def iter_images(root: Union[str, Path]) -> Generator[Path, None, None]:
    for p in Path(root).rglob("*.jpg"):
        yield p
    for p in Path(root).rglob("*.png"):
        yield p


# %%


class ImageFolder(Dataset):
    def __init__(self, root: Union[str, Path], transform = None):
        self.transform = transform
        self.root = root
        self.image_paths = list(iter_images(root))
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, str(self.image_paths[idx])


def load_model(checkpoint):
    if not checkpoint.endswith('.pth'):
        checkpoint += '.pth'

    cp = torch.load(checkpoint)
    classes = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

    model = resnet50(pretrained = False)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 10),
        nn.LogSoftmax(dim = 1),
    )
    model.load_state_dict(cp)
    model = model.to(device)

    return model, classes


def predict(model, dataloader) -> pd.DataFrame:
    model.eval()

    files, preds = [], []
    for x, fn in dataloader:
        x = x.to(device)
        output = model(x)
        pred = torch.argmax(output, 1)
        files += [f for f in fn]
        preds += [p.item() for p in pred]

    result = pd.DataFrame({
        'fn': files,
        'label': preds,
    })
    return result


# %%


def clean_images(root: str):

    transes = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for img_fn in iter_images(root):
        try:
            img = Image.open(img_fn).convert('RGB')
            img = transes(img)
        except Exception as e:
            _ = e
            print(img_fn)
            tgt = img_fn.with_suffix(img_fn.suffix + ".bad")
            img_fn.rename(tgt)


# %%
def predict_dir(
        model,
        classes,
        root: str,
        target: str,
        num_workers ,
        batch_size ,
        clean: bool = False,
):
    if clean:
        clean_images(root)

    if target is None:
        r = Path(root)
        sfx = '.1'
        if r.suffix:
            sfx = r.suffix
            sfx = sfx[:-1] + chr(ord(sfx[-1:]) + 1)
        target = r.with_suffix(sfx)

    # model, classes = load_model('checkpoint.tar')
    data = ImageFolder(root)
    if len(data) == 0:
        return

    dataloader = DataLoader(
        data,
        shuffle = True,
        batch_size = batch_size,
        num_workers = num_workers,
    )
    with torch.no_grad():
        result = predict(model, dataloader)
    result.to_csv(Path(root).name + ".csv", index = False)

    for x in classes:
        Path(target, x).mkdir(0o755, True, True)

    for _, (fn, label) in result.iterrows():
        print(fn, classes[label])
        t = Path(target, classes[label], Path(fn).name)
        Path(fn).rename(t)


def predict_dirs(
        root: str,
        clean: bool = False,
        target: str = None,
        scheme: str = None,
        num_workers: int = 2,
        batch_size: int = 100,
):
    model, classes = load_model(scheme)

    has_processed = False
    for subd in Path(root).iterdir():
        if subd.is_dir():
            has_processed = True
            predict_dir(model, classes, subd, target, num_workers, batch_size, clean)

    if not has_processed:
        predict_dir(model, classes, root, target, num_workers, batch_size, clean)


# %%
# python predict.py --scheme=selfshot --root=test --target=. --clean=0
if __name__ == '__main__':
    Fire(predict_dirs)
