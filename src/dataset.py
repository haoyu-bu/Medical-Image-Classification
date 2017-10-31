import itertools as it
import numpy as np
import tensorflow as tf
import os
import random
import scipy as scp
import scipy.misc

ROOT = os.path.abspath('../')
DATA_PATH = os.path.join(ROOT, 'dataset')

def get_data(batch_size=16):
    print("loading data ... ")

    trn_labels = []
    trn_pixels = []
    tst_labels = []
    tst_pixels = []
    trn = []
    tst = []

    for f in os.listdir(DATA_PATH):
        for fi in os.listdir(os.path.join(DATA_PATH, f)):
            for fii in os.listdir(os.path.join(DATA_PATH, f, fi)):
                for fiii in os.listdir(os.path.join(DATA_PATH, f, fi, fii)):
                    image_path = os.path.join(DATA_PATH, f, fi, fii, fiii)
                    image = scp.misc.imread(image_path, mode='RGB')
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = scp.misc.imresize(image, size=(224, 224))
                    if(fii == "HSIL"):
                        label = int(fiii.split('.')[0])
                    else:
                        label = int(fiii.split('.')[0]) + 3
                    if(f == "trn_data"):
                        trn.append([label, image])
                    else:
                    	tst.append([label, image])

    random.shuffle(trn)
    random.shuffle(tst)
    for i in trn:
        trn_pixels.append(i[1])
        trn_labels.append(i[0])
    for i in tst:
        tst_pixels.append(i[1])
        tst_labels.append(i[0])

    trn_pixels = np.vstack(trn_pixels)
    trn_pixels = trn_pixels.reshape(-1, 3, 32, 32).astype(np.float32)
    
    tst_pixels = np.vstack(tst_pixels)
    tst_pixels = tst_pixels.reshape(-1, 3, 32, 32).astype(np.float32)

    print("-- trn shape = %s" % list(trn_pixels.shape))
    print("-- tst shape = %s" % list(tst_pixels.shape))

    # transpose to tensorflow's bhwc order assuming bchw order
    trn_pixels = trn_pixels.transpose(0, 2, 3, 1)
    tst_pixels = tst_pixels.transpose(0, 2, 3, 1)

    trn_set = batch_iterator(it.cycle(zip(trn_pixels, trn_labels)), batch_size, cycle=True, batch_fn=lambda x: zip(*x))
    tst_set = (tst_pixels, np.array(tst_labels))

    return trn_set, tst_set

def batch_iterator(iterable, size, cycle=False, batch_fn=lambda x: x):
    """
    Iterate over a list or iterator in batches
    """
    batch = []

    # loop to begining upon reaching end of iterable, if cycle flag is set
    if cycle is True:
        iterable = it.cycle(iterable)

    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch_fn(batch)
            batch = []

    if len(batch) > 0:
        yield batch_fn(batch)


if __name__ == '__main__':
    trn, tst = get_data()
