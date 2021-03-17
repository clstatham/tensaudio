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
def custom_he_init(shape, slope=1.0):
    fan_in = np.prod(shape[:-1])
    return np.sqrt(2. / ((1. + slope**2) * fan_in))

#kernel_size = kernel_size + [x.shape.as_list()[3], filters]

#https://github.com/magenta/magenta/blob/c1340b2788af9bc193ef23e1ecec3fabf13d0a14/magenta/models/gansynth/lib/layers.py#L30
def pixel_norm(images, epsilon=1.0e-8):
    a = tf.math.rsqrt(tf.reduce_mean(tf.math.square(images), axis=-1, keepdims=True) + epsilon)
    if tf.reduce_all(tf.math.is_finite(a)):
        return images * a
    else:
        return images

class TAConv1D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, use_bias=False, **kwargs):
        self.filters = filters
        self.kernel_size = [kernel_size]
        self.strides = strides
        self.use_bias = use_bias

        super().__init__(True, 'TAConv1D', **kwargs)
    def build(self, input_shape):
        shape = self.kernel_size + [input_shape[-1], self.filters]
        kernel_scale = custom_he_init(self.kernel_size, 0.0)
        post_scale, init_scale = kernel_scale, 1.0
        self.kernel = self.add_weight(name='kernel', shape=shape, initializer=keras.initializers.random_normal(stddev=kernel_scale), trainable=True)
        super().build(input_shape)

    #@tf.function(experimental_relax_shapes=True)
    def call(self, x):
        return pixel_norm(tf.nn.leaky_relu(K.conv1d(x, kernel=self.kernel, strides=self.strides), 0.2))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)




class SubpixelUpscaling1D(keras.layers.Layer):
    def __init__(self, r):
        super().__init__(trainable=False)
        self.r = r
    #@tf.function(experimental_relax_shapes=True)

    def call(self, x):
        _, _, rc = x.get_shape()
        assert rc % self.r == 0
        #c = rc / r
        y = tf.transpose(x, [2,1,0])
        y = tf.batch_to_space(y, [self.r], [[0,0]])
        y = tf.transpose(y, [2,1,0])
        return y

class TAInitialGeneratorModule(layers.Layer):
    def __init__(self, filters, kernel_size, input_shape):
        super().__init__(True, 'TAInitialGeneratorModule', input_shape=input_shape)
        self.filters = filters
        #self.dns = layers.Dense(input_shape[1])
        self.conv2d_1 = TAConv1D(filters, input_shape[1]-1)
        self.conv2d_2 = TAConv1D(filters, kernel_size)
        #self.bn = layers.BatchNormalization(momentum=0.9)
        #self.pad = layers.ZeroPadding2D((1, input_shape[1]-1))

    #@tf.function(experimental_relax_shapes=True)
    def call(self, x):
        #x = self.dns(x)
        x = tf.expand_dims(x, -1)
        x = pixel_norm(x)
        #x = self.pad(x)
        x = self.conv2d_1(x)
        #x = self.conv2d_2(x)
        return x

class TAGeneratorModule(layers.Layer):
    def __init__(self, filters, kernel_size, scale1=2, scale2=2):
        super().__init__(True, 'TAGeneratorModule')
        self.filters = filters
        self.spu = SubpixelUpscaling1D(scale2)
        self.upscale = layers.UpSampling1D(scale1)
        self.conv_1 = TAConv1D(filters, kernel_size)
        self.conv_2 = TAConv1D(filters, kernel_size*2)
        self.conv_3 = TAConv1D(2, kernel_size)
    
    #@tf.function(experimental_relax_shapes=True)
    def call(self, x):
        x = self.upscale(x)
        x = self.spu(x)
        x = self.conv_1(x)
        x = self.spu(x)
        # x = self.conv_2(x)
        # x = self.spu(x)
        #x = self.conv_3(x)
        return x

class TAFinalGeneratorModule(layers.Layer):
    def __init__(self):
        super().__init__(True, 'TAFinalGeneratorModule')
    
    #@tf.function(experimental_relax_shapes=True)
    def call(self, x):
        x_split = tf.split(x, x.shape[-1]//2, axis=-1)
        y = tf.reduce_mean(tf.stack(x_split, -1), -1)
        # lods = _lods.copy()
        # new_lods = []
        # alphas = K.softmax(K.arange(0.1, 1.0, 1.0/len(lods)))
        # for i, lod in enumerate(lods):
        #     if lod.shape[-1] == lods[-1].shape[-1]:
        #         lods[i] *= alphas[i]
        #         lods[i] = tf.reshape(lods[i], [lods[i].shape[0], -1])
        #         factor = int(np.ceil(2*TOTAL_SAMPLES_OUT / lods[i].shape[-1]))
        #         if factor <= 1:
        #             new_lods += [lods[i][:, :TOTAL_SAMPLES_OUT*2]]
        # y = tf.stack(new_lods, -1)
        # y = tf.reduce_sum(y, -1)
        #y = lods[-1]
        #y = tf.nn.depth_to_space(y, 16)
        #y = tf.reshape(y, [y.shape[0], 2, -1])
        #y = tf.expand_dims(x, 1)
        #y = SubpixelUpscaling1D(y.shape[-1]//2)(y)
        return y


class TAGenerator(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # if GEN_MODE == 0:
        #     #TODO: implement RNN/Hilbert mode
        #     raise NotImplementedError("TODO: implement RNN/Hilbert mode")
        # elif GEN_MODE == 1:
        #     #TODO: port RNN/Audio code to pytorch
        #     #raise NotImplementedError("TODO: implement RNNConv/Audio mode")
        #     self.create_conv_net(mode='rnnaudio')
        # elif GEN_MODE == 2:
        #     self.create_conv_net(mode='hilbert')
        # elif GEN_MODE == 3:
        #     self.create_conv_net(mode='audio')
        # elif GEN_MODE == 4:
        #     self.create_conv_net(mode='mel')
        # elif GEN_MODE == 5:
        #     self.create_conv_net(mode='stft')
        # elif GEN_MODE == 6:
        #     self.create_conv_net(mode='specgram')
        # else:
            # self.total_samp_out = TOTAL_PARAM_UPDATES * N_PARAMS
            # self.desired_process_units = TOTAL_SAMPLES_IN
            # self.create_dense_net(hilb_mode=False)
            # raise ValueError("Created wrong generator class!")
        
        mode = 'specgram'
        self.conv_layers = []
        self.lin_layers = []

        self.sr = SAMPLE_RATE
        # if mode == 'rnnaudio':
        #     self.n_channels = N_CHANNELS
        #     self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels
        #     self.linear_units_out = TOTAL_SAMPLES_IN * GEN_SCALE_LIN * self.n_channels
        #     self.total_samp_out = TOTAL_SAMPLES_OUT
        #     us = TAUpscalingLayer
        #     ds = TADownscalingLayer
        #     bn = BatchRenormalization2D
        #     pool = nn.AdaptiveAvgPool1d
        if mode == 'audio':
            self.n_channels = N_CHANNELS
            self.n_fft = 1
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_INITIAL_LIN_SCALE * self.n_channels
            self.total_samp_out = TOTAL_SAMPLES_OUT
            us = layers.Conv2DTranspose
            ds = layers.Conv2D
            norm = layers.BatchNormalization
            pool = layers.AveragePooling2D
        elif mode == 'mel':
            self.n_channels = 1
            self.n_fft = N_GEN_MEL_CHANNELS
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_fft
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_INITIAL_LIN_SCALE * self.n_fft
            self.total_samp_out = GEN_N_FRAMES
            us = layers.Conv2DTranspose
            ds = layers.Conv2D
            norm = layers.BatchNormalization
            pool = layers.AveragePooling2D
        elif mode == 'stft':
            self.n_channels = 2
            self.n_fft = N_GEN_FFT // 2 + 1
            self.linear_units_in = TOTAL_SAMPLES_IN * 2 * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_INITIAL_LIN_SCALE * self.n_channels
            self.initial_n_fft = self.linear_units_out // 2
            self.total_samp_out = int(self.n_channels * self.n_fft * GEN_N_FRAMES * 1)
            us = layers.Conv2DTranspose
            ds = layers.Conv2D
            norm = layers.BatchNormalization
            pool = layers.AveragePooling2D
        elif mode == 'specgram':
            self.n_channels = 2
            self.n_fft = 1
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_INITIAL_LIN_SCALE * self.n_channels
            self.total_samp_out = int(TOTAL_SAMPLES_OUT * self.n_channels * 1)
            self.initial_n_fft = 1
            us = layers.Conv1DTranspose
            ds = layers.Conv1D
            norm = layers.BatchNormalization
            pool = layers.AveragePooling1D
        dns = layers.Dense
        self.is_2d = ds.__name__.find('2D') != -1

        self.initializer = keras.initializers.HeNormal()

        self.use_bias = False

        v_cprint("*-"*39 + "*")
        v_cprint("Target length is", self.total_samp_out, "samples.")

        dummy = tf.Variable(generate_input_noise(), trainable=False)

        


        self.n_sets = 0
        self.max_sets = 4
        self.base_scale = 2
        self.max_scale = 4
        
        prob = 0.01
        s_us = 2
        k_us = 3
        k_div = 8
        s_ds = 2
        k_ds = 2

        self.dim = 16
        dim_mul = 2

        self.net = [
            dns(self.n_channels*self.dim*dim_mul, use_bias=False, input_shape=[TOTAL_SAMPLES_IN]),
            #dns(TOTAL_SAMPLES_OUT, use_bias=False, input_shape=[TOTAL_SAMPLES_IN]),
        ]
        dummy = self.net[-1](dummy)

        #lods = []
        self.net.append(TAInitialGeneratorModule(self.dim, k_us, dummy.shape))
        dummy = self.net[-1](dummy)
        while dummy.shape[-2] < TOTAL_SAMPLES_OUT:
        #while self.n_sets < self.max_sets:
            self.net.append(TAGeneratorModule(dummy.shape[-1]*dim_mul, k_us, self.get_scale(self.n_sets), 2))
            #self.net.append(TAGeneratorModule(dummy.shape[-1]*dim_mul, k_us, 2, 2))
            dummy = self.net[-1](dummy)
            pool_scale = dummy.shape[-2]//TOTAL_SAMPLES_OUT
            if pool_scale > 1:
                self.net.append(layers.AvgPool1D(k_us, pool_scale, padding='valid'))
                dummy = self.net[-1](dummy)

            # self.net.append(us(dummy.shape[-1]*dim_mul, k_us, s_us, use_bias=False, padding='same', kernel_initializer=self.initializer, trainable=False)) # padding_mode = 'reflect'
            # dummy = self.net[-1](dummy)
            # self.net.append(norm(momentum=0.9))
            # dummy = self.net[-1](dummy)
            # self.net.append(layers.ReLU())
            # dummy = self.net[-1](dummy)
            # self.net.append(SubpixelUpscaling1D(dummy.shape[-1]//self.n_channels))
            # dummy = self.net[-1](dummy)
            # while tf.size(dummy) // dummy.shape[0] > self.total_samp_out * 2:
            #     self.net.append(layers.AveragePooling1D(2, 2, trainable=False, padding='valid'))
            #     dummy = self.net[-1](dummy)
            #     self.net.append(norm(momentum=0.9))
            #     dummy = self.net[-1](dummy)
            #     self.net.append(layers.Activation(tf.nn.tanh))
            #     dummy = self.net[-1](dummy)

            # self.net.append(layers.Lambda(lambda x: x))
            # dummy = self.net[-1](dummy)
            #lods.append(dummy)

            # dim_mul //= 2
            # dim_mul = max(dim_mul, 2)
            # k_us //= dim_mul
            # k_us = max(k_us, 1)
            self.n_sets += 1
        print("Created", self.n_sets, "sets of Generator layers.")
        
        self.final_layer = TAFinalGeneratorModule()
        dummy = self.final_layer(dummy)
        v_cprint("Created final layers.")
        v_cprint("Final shape:", list(dummy.shape))

        # if self.is_2d:
        #     dummy = torch.cat((dummy, torch.zeros((dummy.shape[0], dummy.shape[1], 1, dummy.shape[3])).to(dummy.device)), -2)
        #self.net = keras.Sequential(self.net)
        for layer in self.net:
            if not layer.trainable:
                layer.trainable = True
        #self.net.call(generate_input_noise())
        #self.net.summary()
        #self.summary()

    def get_scale(self, step):
        return min(self.max_scale, self.base_scale ** step)

    def sanity_check(self, data):
        # if torch.isnan(data).any():
        #     raise RuntimeError("Data is NaN!")
        pass

    #@tf.function(experimental_relax_shapes=True)
    def run_conv_net(self, inp, mode, training):
        #print(inp.min().item(), inp.max().item(), inp.mean().item())
        data = normalize_negone_one(tf.cast(inp, tf.float32))
        #print(data.min().item(), data.max().item(), data.mean().item())
        #vv_cprint("|}} Initial data.shape:", data.shape)

        self.sanity_check(data)

        lods = []
        for i, layer in enumerate(self.net):
            data = layer(data, training=training)
            # if layer.__class__.__name__.find('Lambda') != -1:
            #     lods.append(data)
        post_final = self.final_layer(data)

        self.sanity_check(post_final)
        if mode == 'mel' and DIS_MODE == 2:
            # ret = check_and_fix_size(data.squeeze(), self.n_fft, -2)
            raise NotImplementedError
        elif mode == 'mel':
            # invmel_inp = check_and_fix_size(data, self.n_fft, -2)
            # stft = MelToSTFTWithGradients.apply(invmel_inp, self.n_fft)
            # data = stft_to_audio(stft, GEN_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_PREVIEW)
            raise NotImplementedError
        elif mode == 'stft':
            #pre_stftgram = post_final.reshape(batch_sz, self.n_channels, self.n_fft, -1).requires_grad_(True)#[..., :self.total_samp_out]
            # pre_stftgram = post_final[..., :GEN_N_FRAMES]
            # if torch.is_grad_enabled():
            #     pre_stftgram.retain_grad()
            # stftgram = torch.stack((normalize_zero_one(pre_stftgram[:,0]), normalize_negone_one(pre_stftgram[:,1])), dim=1)
            # if torch.is_grad_enabled():
            #     stftgram.retain_grad()
            
            # ret = normalize_negone_one(stftgram_to_audio(stftgram, GEN_HOP_LEN))
            # self.sanity_check(ret)
 
            # out = ret.flatten(0)
            # #out = ret[0]
            # write_normalized_audio_to_disk(out, "./test1.wav")
            raise NotImplementedError
        elif mode == 'specgram':
            #pre_specgram = tf.reshape(post_final, [tf.shape(post_final)[0], -1, self.n_channels])
            pre_specgram = post_final

            specgram = tf.stack([
                normalize_zero_one(pre_specgram[:,:,0]),
                linear_to_mel(normalize_negone_one(pre_specgram[:,:,1]))
                ], axis=-1)
            

            self.sanity_check(specgram)
            ret = specgram_to_audio(tf.squeeze(specgram))
            self.sanity_check(ret)

            # print("Pre-scaled Specgram:")
            # print("Mag min/max/mean:", pre_specgram[:,0].min(), pre_specgram[:,0].max(), pre_specgram[:,0].mean())
            # print("Phase min/max/mean:", pre_specgram[:,1].min(), pre_specgram[:,1].max(), pre_specgram[:,1].mean())
            # print("Post-scaled Specgram:")
            # print("Mag min/max/mean:", specgram[0,0].min().item(), specgram[0,0].max().item(), specgram[0,0].mean().item())
            # print("Phase min/max/mean:", specgram[0,1].min().item(), specgram[0,1].max().item(), specgram[0,1].mean().item())
            #out = ret.flatten(0)
            #out = ret[0]
            #write_normalized_audio_to_disk(out, "./test1.wav")
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # ax1.plot(specgram[0,0].contiguous().detach().cpu().numpy())
            # ax2.plot(specgram[0,1].contiguous().detach().cpu().numpy())
            # plt.show()
        else:
            raise NotImplementedError
        
        return ret
    
    #@tf.function(experimental_relax_shapes=True)
    def gen_fn(self, inputs, training=True):
        return self.run_conv_net(inputs, 'specgram', training)[:, :TOTAL_SAMPLES_OUT]
    
    #@tf.function(experimental_relax_shapes=True)
    def call(self, inputs, mode):
        return self.gen_fn(inputs, mode == tf.estimator.ModeKeys.TRAIN)