import keras
import numpy as np
from keras.models import load_model
import argparse
import scipy as scp
import scipy.misc
import os

img_rows = 224
img_cols = 224
img_channel = 3

mu = [[148.47410684, 103.53716248, 100.33421473], [146.0697635, 103.85407198, 101.29443072], [125.4883176, 77.20283499, 54.2456892]]
std = [[45.57637533, 38.05719382, 38.02870385], [46.10150803, 37.30163753, 37.11030376], [65.29455049, 54.18454717, 48.43651913]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", help="path to weights file")
    parser.add_argument("image_dir", help="path to image")
    args = parser.parse_args()

    # load data
    x_test = []
    for i in os.listdir(args.image_dir):
        image = scp.misc.imread(os.path.join(args.image_dir, i), mode='RGB')
        image = scp.misc.imresize(image, size=(img_rows, img_cols))
        x_test.append(image)

    # preprocess
    for i in range(3):
        x_test[i] = (x_test[i] - mu[i]) / std[i]

    # build network
    model = load_model(args.weights) 
    print(model.summary())

    print(np.argmax(model.predict(x_test)))