import six
import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import matplotlib
import matplotlib.pyplot as plt
import librosa

from helper import *
from hilbert import *
from global_constants import *

# def pixelshuffle1d(x, upscale_factor):
#     batch_size = x.shape[0]
#     short_channel_len = x.shape[1]
#     short_width = x.shape[2]

#     long_channel_len = short_channel_len // upscale_factor
#     long_width = upscale_factor * short_width

#     x = x.contiguous().view([batch_size, upscale_factor, long_channel_len, short_width])
#     x = x.permute(0, 2, 3, 1).contiguous()
#     x = x.view(batch_size, long_channel_len, long_width)

#     return x

# https://github.com/magenta/magenta/blob/c1340b2788af9bc193ef23e1ecec3fabf13d0a14/magenta/models/gansynth/lib/layers.py#L30


def pixel_norm(images, epsilon=1.0e-8):
    a = tf.math.rsqrt(tf.reduce_mean(tf.math.square(
        images), axis=-1, keepdims=True) + epsilon)
    if tf.reduce_all(tf.math.is_finite(a)):
        return images * a
    else:
        return images


class TAConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, groups=1, strides=1, use_bias=True, **kwargs):
        super().__init__()
        # self.filters = filters
        # self.kernel_size = [kernel_size]
        # self.strides = strides
        # self.use_bias = use_bias
        self.init = keras.initializers.HeNormal()
        self.conv = layers.Conv1D(
            filters, kernel_size, strides,
            use_bias=use_bias,
            groups=groups,
            padding="same",
            kernel_initializer=self.init,
            name='TAConv1D', **kwargs
        )

    def call(self, x):
        return pixel_norm(tf.nn.leaky_relu(self.conv(x), 0.2))


class SubpixelUpscaling1D(keras.layers.Layer):
    def __init__(self, r):
        super().__init__(trainable=False)
        self.r = r

    def call(self, x):
        _, _, rc = x.get_shape()
        assert rc % self.r == 0
        #c = rc / r
        y = tf.transpose(x, [2, 1, 0])
        y = tf.batch_to_space(y, [self.r], [[0, 0]])
        y = tf.transpose(y, [2, 1, 0])
        return y


class TAInitialGeneratorModule(layers.Layer):
    def __init__(self, filters, kernel_size, input_shape):
        super().__init__(True, 'TAInitialGeneratorModule', input_shape=input_shape)
        self.filters = filters
        self.conv2d_1 = TAConv1D(filters, input_shape[1]-1, groups=1)

    def call(self, x):
        x = tf.expand_dims(x, -1)
        x = pixel_norm(x)
        x = self.conv2d_1(x)
        return x


class TAGeneratorModule(layers.Layer):
    def __init__(self, filters, kernel_size, idx, scale1=2, scale2=2):
        super().__init__(True, 'TAGeneratorModule'+str(idx))
        self.filters = filters
        self.spu = SubpixelUpscaling1D(scale2)
        self.upscale = layers.UpSampling1D(scale1)
        self.conv_1 = TAConv1D(filters, kernel_size, groups=16)

    def call(self, x):
        x = self.upscale(x)
        x = self.conv_1(x)
        x = self.spu(x)
        return x


class TAFinalGeneratorModule(layers.Layer):
    def __init__(self):
        super().__init__(True, 'TAFinalGeneratorModule')

    def call(self, x):
        x_split = tf.split(x, x.shape[-1]//2, axis=-1)
        y = tf.reduce_mean(tf.stack(x_split, -1), -1)
        return y


def create_generator(dim=16, base_scale=2, max_scale=2):
    def gen_scale(step):
        return min(max_scale, base_scale ** step)

    initializer = keras.initializers.HeNormal()

    v_cprint("*-"*39 + "*")
    v_cprint("Target length is", TOTAL_SAMPLES_OUT, "samples.")

    dummy = tf.Variable(generate_input_noise(), trainable=False)

    n_sets = 0

    k_us = 333
    dim_mul = 16

    net = [
        layers.Dense(2*dim*dim_mul, input_shape=[TOTAL_SAMPLES_IN]),
    ]
    dummy = net[-1](dummy)

    #lods = []
    net.append(TAInitialGeneratorModule(dim, 3, dummy.shape))
    dummy = net[-1](dummy)
    while dummy.shape[-2] < TOTAL_SAMPLES_OUT:
        net.append(TAGeneratorModule(dim*dim_mul, k_us,
                   idx=n_sets, scale1=gen_scale(n_sets), scale2=2))
        dummy = net[-1](dummy)

        pool_scale = dummy.shape[-2]//TOTAL_SAMPLES_OUT
        if pool_scale > 1:
            net.append(layers.AvgPool1D(k_us, pool_scale, padding='valid'))
            dummy = net[-1](dummy)

        dim_mul //= 2
        dim_mul = max(dim_mul, 2)
        n_sets += 1
    print("Created", n_sets, "sets of Generator layers.")

    net.append(TAFinalGeneratorModule())
    dummy = net[-1](dummy)
    v_cprint("Generator output shape:", list(dummy.shape))

    net = keras.Sequential(net)
    for layer in net.layers:
        if not layer.trainable:
            layer.trainable = True
    return net


def generator(gen_net, data, training):
    post_net = gen_net(data, training=training)

    specgram = tf.stack([
        K.sigmoid(post_net[:, :, 0]),
        linear_to_mel(K.tanh(post_net[:, :, 1]))
    ], axis=-1)

    ret = specgram_to_audio(tf.squeeze(specgram))

    return ret[:, :TOTAL_SAMPLES_OUT]
