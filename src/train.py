import keras
import math
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, add, Activation
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint, Callback
from keras.models import Model
from keras import optimizers
from keras.utils import plot_model
import dataset_pca
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import densenet
import resnext
import resnet
import vgg19

img_rows, img_cols = 224,224 
img_channels       = 3
num_classes        = 2 
batch_size         = 32        # 64 or 32 or other
epochs             = 130
iterations         = 120      

os.environ["CUDA_VISIBLE_DEVICES"]='2'

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

def scheduler(epoch):
    if epoch <= 80:
        return 0.1
    if epoch <= 120:
        return 0.01
    if epoch <= 180:
        return 0.001
    return 0.0005

def data_generator(x_train, y_train, batchsize=32):
    datagen1   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
    datagen2   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
    datagen3   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen1.fit(x_train[0])
    datagen2.fit(x_train[1])
    datagen3.fit(x_train[2])
    dg_1 = datagen1.flow(x_train[0], y_train, batchsize, shuffle=False)
    dg_2 = datagen2.flow(x_train[1], y_train, batchsize, shuffle=False)
    dg_3 = datagen3.flow(x_train[2], y_train, batchsize, shuffle=False)
    while 1: 
        batch_1 = next(dg_1)        
        batch_2 = next(dg_2)        
        batch_3 = next(dg_3)
        yield ([batch_1[0], batch_2[0], batch_3[0]], batch_1[1])

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls_pos = []
        self.val_recalls_neg = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.argmax((np.asarray(self.model.predict(
            [self.validation_data[0], self.validation_data[1], self.validation_data[2]]))), axis=1)
        val_targ = np.argmax(self.validation_data[3], axis=1)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall_pos = recall_score(val_targ, val_predict, pos_label=1)
        _val_recall_neg = recall_score(val_targ, val_predict, pos_label=0)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls_pos.append(_val_recall_pos)
        self.val_recalls_neg.append(_val_recall_neg)
        self.val_precisions.append(_val_precision)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("net", help="net used")
    parser.add_argument("save_dir", help="save directory")
    args = parser.parse_args()


    # load data
    (x_train, y_train), (x_test, y_test) = dataset_pca.get_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)


    for j in range(3):
        x_train[j] = np.vstack(x_train[j]).reshape(-1,img_rows,img_cols,3).astype('float32')
        x_test[j] = np.vstack(x_test[j]).reshape(-1,img_rows,img_cols,3).astype('float32')

    # build network
    img_input_1 = Input(shape=(img_rows,img_cols,img_channels))
    img_input_2 = Input(shape=(img_rows,img_cols,img_channels))
    img_input_3 = Input(shape=(img_rows,img_cols,img_channels))

    if(args.net == "densenet"):
        output_1    = densenet.densenet(img_input_1,num_classes)
        output_2    = densenet.densenet(img_input_2,num_classes)
        output_3    = densenet.densenet(img_input_3,num_classes)
    elif(args.net == "resnet"):
        output_1    = resnet.resnet(img_input_1,num_classes)
        output_2    = resnet.resnet(img_input_2,num_classes)
        output_3    = resnet.resnet(img_input_3,num_classes)
    elif(args.net == "resnext"):
        output_1    = resnext.resnext(img_input_1,num_classes)
        output_2    = resnext.resnext(img_input_2,num_classes)
        output_3    = resnext.resnext(img_input_3,num_classes)
    elif(args.net == "vgg19"):
        output_1    = vgg19.vgg19(img_input_1,num_classes)
        output_2    = vgg19.vgg19(img_input_2,num_classes)
        output_3    = vgg19.vgg19(img_input_3,num_classes)
    output = keras.layers.Add()([output_1, output_2, output_3, output_3, output_2, output_3])
    output = Activation('softmax')(output)
    model     = Model([img_input_1, img_input_2, img_input_3], output)
    #model.load_weights('ckpt.h5')
    
    plot_model(model, show_shapes=True, to_file='model.png')
    print(model.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    metrics = Metrics()
    tb_cb     = TensorBoard(log_dir=args.save_dir, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    ckpt      = ModelCheckpoint(os.join(args.save_dir, './ckpt.h5'), save_best_only=False, mode='auto', period=10)
    cbks      = [change_lr,tb_cb,ckpt,metrics]

    # set data augmentation
    print('Using real-time data augmentation.')
    data_generator = data_generator(x_train, y_train, batch_size)

    # start training
    model.fit_generator(data_generator, steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=([x_test[0], x_test[1], x_test[2]], y_test))
    for i in range(epochs):
        print "epoch" + str(i)
        print(metrics.val_recalls_pos)
        print(metrics.val_recalls_neg)
        print(metrics.val_precisions)
        print(metrics.val_f1s)
    model.save(os.join(args.save_dir, 'densenet.h5'))
