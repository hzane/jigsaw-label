from itertools import cycle
import tkinter as tk
import os
import pandas as pd
from pathlib import Path
from PIL import Image, ImageTk


class slideshow_app(tk.Tk):
    def __init__(self, images, x, y):
        tk.Tk.__init__(self)
        self.geometry('+{}+{}'.format(x, y))
        self.picture_canvas = tk.Canvas(self, bd = 0, highlightthickness = 0)
        self.bind("<ButtonPress-1>", self.show_slides)
        self.bind("<Key>", self.show_slides)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("q", lambda e: self.destroy())
        self.picture_canvas.pack()
        # self.pictures = cycle(
        #     (ImageTk.PhotoImage(file = fn), desc) for fn, desc in images
        # )
        self.pictures = images
        self.picture_display = tk.Label(self.picture_canvas, bd = 0)
        self.picture_display.pack(side = tk.LEFT)

        self.tag = tk.Label(
            self.picture_canvas,
            font = ('Sarasa Mono SC', 16),
            justify = tk.LEFT,
        )
        self.tag.pack(side = tk.LEFT, anchor = 's')
        # self.tag.place(x = 30, y = 30)

    def show_slides(self, event = None):
        imgfn, imgdesc = next(self.pictures)
        img = ImageTk.PhotoImage(file=imgfn)
        self.picture_display.config(image = img, width=min(img.width(), 2000), height =min(img.height(), 1400),)
        self.tag.config(text = imgdesc)
        self.title(imgfn)

    def run(self):
        self.mainloop()


def ltrim(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]


def ltrims(text, *prefixes):
    for prefix in prefixes:
        text = ltrim(text, prefix)
    return text


def images(meta: str):
    data = pd.read_json(meta, lines = True)
    data.rename(columns = lambda n: ltrims(n, 'v.tag.', 'tag.', 'fuzzy.'), inplace = True)

    data = data.sample(frac = 1.).reset_index(drop = True)
    for _, row in data.iterrows():
        uri = row.uri
        row = row[~row.isna()]
        des = ''.join(
            f'{name}:\t{val:.1f}\n' for name, val in row.drop(index = 'uri').iteritems()
        )
        yield uri, des


def slideshow(meta: str = 'nselfshot-tx.json'):
    x = 100
    y = 50

    app = slideshow_app(images(meta), x, y)
    app.show_slides()
    app.run()


def nima_show(meta: str):
    dataset = pd.read_csv(
        meta, sep = '\t', header = None, names = ['uri', 'score', 'istd']
    )

    def iter_image(dataset):
        for _, row in dataset.iterrows():
            yield row.uri, f'score: {row.score:.3f}, std: {row.istd:.3f}'

    app = slideshow_app(iter_image(dataset), 100, 50)
    app.show_slides()
    app.run()
    pass


if __name__ == '__main__':
    from fire import Fire
    Fire()
    # slideshow(meta)
