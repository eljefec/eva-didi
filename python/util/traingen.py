from __future__ import print_function

import numpy as np
import os

def get_size(train_dir, index_file):
    fullpath = os.path.join(train_dir, index_file)
    ids = []
    with open(fullpath, 'r') as f:
        for id in f:
            ids.append(id)
    return len(ids)

def get_example_path(dir, id, ext):
    return os.path.join(dir, '{:06d}.{}'.format(id, ext))

def write_train_val(outdir, count):
    idx = []
    with open(os.path.join(outdir, 'trainval.txt'), 'w') as f:
        for i in range(count):
            formatted = '{:06d}'.format(i)
            print(formatted, file=f)
            idx.append(formatted)

    train_file = os.path.join(outdir, 'train.txt')
    val_file = os.path.join(outdir, 'val.txt')

    idx = np.random.permutation(idx)

    val_split = len(idx) / 4
    val_idx = sorted(idx[:val_split])
    train_idx = sorted(idx[val_split:])

    with open(train_file, 'w') as f:
        for i in train_idx:
            print(i, file=f)

    with open(val_file, 'w') as f:
        for i in val_idx:
            print(i, file=f)

    print('Training set is saved to ' + train_file)
    print('Validation set is saved to ' + val_file)
