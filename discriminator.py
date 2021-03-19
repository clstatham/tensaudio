import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import numpy as np
#from online_norm_pytorch import OnlineNorm1d, OnlineNorm2d
from helper import *
from global_constants import *
from hilbert import *
import os
import inspect

class ClipConstraint(keras.constraints.Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)
    
    def get_config(self):
        return {'clip_value': self.clip_value}

def create_discriminator():
    x = TOTAL_SAMPLES_OUT * 2
    
    initializer = keras.initializers.HeNormal()
    # const = ClipConstraint(0.01)

    net = []
    i = 0
    c = 2 * 2
    n = 2
    k1 = DIS_KERNEL_SIZE
    s1 = DIS_STRIDE
    k2 = DIS_KERNEL_SIZE
    s2 = DIS_STRIDE
    while x > 256:
        c = min(DIS_MAX_CHANNELS, int(2 * 2**i))
        n = min(DIS_MAX_CHANNELS, int(2 * 2**(i+1)))
        if x <= DIS_KERNEL_SIZE:
            s1 = 1
            k1 = 1
        
        i += 1
        x = (x - (k1 - 1) - 1) // s1 + 1
        net.append(layers.Conv1D(n, k1, s1, kernel_initializer=initializer))
        #self.net.append(layers.BatchNormalization())
        #self.net.append(layers.Dropout(DIS_DROPOUT))
        net.append(layers.LeakyReLU(0.2))
    print("Created", i, "sets of Discriminator Conv layers.")
    net.append(layers.Flatten())
    net.append(layers.Dense(128))
    net.append(layers.Dense(64))
    net.append(layers.Dense(32))
    net.append(layers.Dense(16))
    net.append(layers.Dense(8))
    net.append(layers.Dense(1))

    net = keras.Sequential(net)
    net.build([None, TOTAL_SAMPLES_OUT, 2])
    return net
    
def discriminator(dis_net, data, training=True):
    data = audio_to_specgram(data)

    verdicts = dis_net(data, training=training)
    
    return tf.squeeze(verdicts)
