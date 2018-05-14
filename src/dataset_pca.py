import itertools as it
import numpy as np
import tensorflow as tf
import os
import random
import scipy as scp
import scipy.misc
from sklearn.decomposition import PCA


ROOT = os.path.abspath('../../../')
DATA_PATH = os.path.join(ROOT, 'dataset')
img_rows = 32
img_cols = 32
img_channel = 3

def get_data(batch_size=16):
    print("loading data ... ")

    trn_labels = []
    trn_pixels = []
    trn_1 = []
    trn_2 = []
    trn_3 = []

    tst_labels = []
    tst_pixels = []
    tst_1 = []
    tst_2 = []
    tst_3 = []

    trn_ele = []
    tst_ele = []

    for f in os.listdir(DATA_PATH):
        for fi in os.listdir(os.path.join(DATA_PATH, f)):
            if(fi == "HSIL"):
                #label = int(fiii.split('.')[0]) - 1
                label = 0
            elif(fi == "Normal"):
                label = 1
            else:
                continue
                #label = int(fiii.split('.')[0]) + 2
            for fii in os.listdir(os.path.join(DATA_PATH, f, fi)):
                for fiii in os.listdir(os.path.join(DATA_PATH, f, fi, fii)):
                    image_path = os.path.join(DATA_PATH, f, fi, fii, fiii)
                    image = scp.misc.imread(image_path, mode='RGB')
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = scp.misc.imresize(image, size=(img_rows, img_cols))    
                    '''
                    # pca
                    pca = PCA(n_components=32)
                    img = np.zeros((img_rows, img_cols, img_channel))
                    for c in range(img_channel):
                        tmp = pca.fit_transform(image[:,:,c])
                        tmp = np.transpose(tmp)
                        tmp = pca.fit_transform(tmp)
                        img[:,:,c] = np.transpose(tmp)
                    image = img
                    # print(np.shape(image))
                    '''         
                    
                    if(fiii == "1.jpg"):
                        if(f == "trn_data"):
                            trn1 = image
                        else:
                            tst1 = image
                    elif(fiii == "2.jpg"):
                        if(f == "trn_data"):
                            trn2 = image
                        else:
                            tst2 = image
                    elif(fiii == "3.jpg"):
                        if(f == "trn_data"):
                            trn3 = image
                        else:
                            tst3 = image

                if(f == "trn_data"):  
                    trn_ele.append([trn1,trn2,trn3,label])
                elif(f == "tst_data"):   
                    tst_ele.append([tst1,tst2,tst3,label])


    random.shuffle(trn_ele)
    random.shuffle(tst_ele)

    for i in trn_ele:
        trn_1.append(i[0])
        trn_2.append(i[1])
        trn_3.append(i[2])
        trn_labels.append(i[3])

    trn_pixels.append(trn_1)
    trn_pixels.append(trn_2)
    trn_pixels.append(trn_3)

    '''
    for i in trn_pixels:
        i = np.vstack(i)
        i = i.reshape(-1, img_channel, img_rows, img_cols).astype(np.float32)
        # transpose to tensorflow's bhwc order assuming bchw order
        i = i.transpose(0, 2, 3, 1)
    #trn_pixels = np.vstack(trn_pixels)
    '''

    for i in tst_ele:
        tst_1.append(i[0])
        tst_2.append(i[1])
        tst_3.append(i[2])
        tst_labels.append(i[3])

    tst_pixels.append(tst_1)
    tst_pixels.append(tst_2)
    tst_pixels.append(tst_3)
    '''
    for i in tst_pixels:
        i = np.vstack(i)
        i = i.reshape(-1, img_channel, img_rows, img_cols).astype(np.float32)
        # transpose to tensorflow's bhwc order assuming bchw order
        i = i.transpose(0, 2, 3, 1)
    #tst_pixels = np.vstack(tst_pixels)
    

    # pca
    pca = PCA(n_components=32*32*3)
    for i in range(3):
            imgs_1d = (trn_pixels[i] + tst_pixels[i])[:,:,:,c].flatten()
            tmp = pca.fit_transform()
    '''
    # mean normalization
    for i in range(3):
        mu = np.average(trn_pixels[i] + tst_pixels[i], axis=(0, 1, 2))
        std = np.std(trn_pixels[i] + tst_pixels[i], axis=(0, 1, 2))
        trn_pixels[i] = (trn_pixels[i] - mu) / std
        


    print("-- trn shape = %s" % list(np.shape(trn_pixels)))
    print("-- tst shape = %s" % list(np.shape(tst_pixels)))

    #trn_set = batch_iterator(it.cycle(zip(trn_pixels, trn_labels)), batch_size, cycle=True, batch_fn=lambda x: zip(*x))
    #tst_set = batch_iterator(it.cycle(zip(tst_pixels, tst_labels)), batch_size, cycle=True, batch_fn=lambda x: zip(*x))
    #tst_set = (tst_pixels, np.array(tst_labels))

    return (trn_pixels, trn_labels), (tst_pixels, tst_labels)

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
    (x_tr, y_tr), (x_ts, t_ts) = get_data()
