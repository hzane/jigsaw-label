#!/usr/bin/env python

import torch

from fire import Fire


def reset(scheme, origin:float=0.5):
    fn = scheme+'.tar'
    cp = torch.load(fn)
    prev = cp['acc']
    cp['acc'] = origin
    torch.save(cp, fn)
    print(f'change best-acc from {prev:.3f} to {origin:.3f}')


if __name__ == '__main__':
    Fire(reset)
