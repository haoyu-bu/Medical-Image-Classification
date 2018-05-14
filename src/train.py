import keras
import math
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras.utils import plot_model
import dataset
import dataset_pca
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

growth_rate        = 12 
depth              = 100
compression        = 0.5

img_rows, img_cols = 32, 32
img_channels       = 3
num_classes        = 2 
batch_size         = 16         # 64 or 32 or other
epochs             = 300
iterations         = 100       
weight_decay       = 0.0001


os.environ["CUDA_VISIBLE_DEVICES"]='3'


config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

KTF.set_session(sess)


def scheduler(epoch):
    if epoch <= 40:
        return 0.1
    if epoch <= 150:
        return 0.01
    if epoch <= 250:
        return 0.001
    return 0.0005

def densenet(img_input,classes_num):

    def bn_relu(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        x = bn_relu(x)
        x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        return x

    def single(x):
        x = bn_relu(x)
        x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x,blocks,nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    def dense_layer(x):
        return Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)


    nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2

    x = Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(img_input)

    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x

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

if __name__ == '__main__':

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
    output_1    = densenet(img_input_1,num_classes)
    output_2    = densenet(img_input_2,num_classes)
    output_3    = densenet(img_input_3,num_classes)
    output = keras.layers.Average()([output_1, output_2, output_3]) 
    model     = Model([img_input_1, img_input_2, img_input_3], output)
    #model.load_weights('densenet.h5')
    
    plot_model(model, show_shapes=True, to_file='model.png')
    print(model.summary())

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb     = TensorBoard(log_dir='./densenet/', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    ckpt      = ModelCheckpoint('./ckpt.h5', save_best_only=False, mode='auto', period=10)
    cbks      = [change_lr,tb_cb,ckpt]

    # set data augmentation
    print('Using real-time data augmentation.')
    data_generator = data_generator(x_train, y_train, batch_size)

    # start training
    model.fit_generator(data_generator, steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=([x_test[0], x_test[1], x_test[2]], y_test))
    model.save('densenet.h5')
