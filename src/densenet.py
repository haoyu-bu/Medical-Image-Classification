import keras
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras import optimizers
from keras import regularizers

growth_rate        = 12 
depth              = 169
compression        = 0.5
weight_decay       = 0.0001

def densenet(img_input, classes_num):

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


    #nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2
    if depth == 121:
        stages = [6, 12, 24, 16] 
    elif depth == 169:
        stages = [6, 12, 32, 32]  
    elif depth == 201:
        stages = [6, 12, 48, 32]  
    elif depth == 161:
        stages = [6, 12, 36, 24]  
 
    x = Conv2D(nchannels,kernel_size=(7,7),strides=(2,2),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(img_input)
    x = bn_relu(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x, nchannels = dense_block(x,stages[0],nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,stages[1],nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,stages[2],nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,stages[3],nchannels)
    x, nchannels = transition(x,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x
