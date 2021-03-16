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

class SubpixelUpscaling1D(keras.layers.Layer):
    def __init__(self, r):
        super().__init__(trainable=False)
        self.r = r
    @tf.function
    def call(self, x):
        _, _, rc = x.get_shape()
        assert rc % self.r == 0
        #c = rc / r
        y = tf.transpose(x, [2,1,0])
        y = tf.batch_to_space(y, [self.r], [[0,0]])
        y = tf.transpose(y, [2,1,0])
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

        self.dim = 16
        dim_mul = 16

        self.net = [
            dns(self.n_channels*self.dim*dim_mul, use_bias=False, input_shape=[TOTAL_SAMPLES_IN]),
        ]
        dummy = self.net[0](dummy)
        #v_cprint("Created Linear layer.", self.initial_layers.input_shape, ">", self.initial_layers.output_shape)
        self.net.append(layers.Dropout(GEN_DROPOUT))
        dummy = self.net[-1](dummy)


        self.n_sets = 0
        self.blur_amount = 1
        
        prob = 0.01
        s_us = 2
        k_us = 444
        k_div = 8
        s_ds = 2
        k_ds = 2
        
        

        rnn_index = 0

        if self.is_2d:
            dummy = tf.reshape(dummy, [dummy.shape[0], self.dim, 4, -1])
        else:
            #dummy = tf.reshape(dummy, [dummy.shape[0], self.dim, -1])
            self.net.append(layers.Reshape([self.dim, -1]))
            dummy = self.net[-1](dummy)
        samps_post_ds = tf.size(dummy)

        while tf.size(dummy) // dummy.shape[0] < self.total_samp_out:
            self.net.append(us(dummy.shape[-1]*dim_mul, k_us, s_us, use_bias=False, padding='same', kernel_initializer=self.initializer, trainable=False)) # padding_mode = 'reflect'
            dummy = self.net[-1](dummy)
            self.net.append(norm(momentum=0.9))
            dummy = self.net[-1](dummy)
            self.net.append(layers.ReLU())
            dummy = self.net[-1](dummy)
            self.net.append(SubpixelUpscaling1D(dummy.shape[-1]//self.n_channels))
            dummy = self.net[-1](dummy)
            while tf.size(dummy) // dummy.shape[0] > self.total_samp_out * 2:
                self.net.append(ds(dummy.shape[-1], k_us, 2, use_bias=False, kernel_initializer=self.initializer, trainable=False))
                dummy = self.net[-1](dummy)
                self.net.append(norm(momentum=0.9))
                dummy = self.net[-1](dummy)
                self.net.append(layers.Activation(tf.nn.tanh))
                dummy = self.net[-1](dummy)
            dim_mul //= 2
            dim_mul = max(dim_mul, 1)
            k_us //= dim_mul
            k_us = max(k_us, 1)
            self.n_sets += 1


        """
        n = 1
        while n**2 < GEN_MAX_CHANNELS:
            self.conv_layers.append(nn.Conv1d(n, n*2, 1, 1, groups=min(2,n), bias=False))
            dummy = self.conv_layers[-1](dummy)
            n *= 2
        print("Created {} Convolution layers until n={}.".format(len(self.conv_layers), n))
        
        while samps_post_ds < self.total_samp_out or self.n_sets < GEN_MIN_LAYERS:
            self.n_sets += 1
            #k_us = GEN_KERNEL_SIZE_UPSCALING * self.n_sets + 1
            #k_ds = GEN_KERNEL_SIZE_DOWNSCALING * max(self.n_sets // 25, 1)
            #n = min(GEN_MAX_CHANNELS, (2**(self.n_sets)))
            c = min(n**2, GEN_MAX_CHANNELS)
            v_cprint("="*80)

            # if samps_post_ds >= self.total_samp_out:
            #     s_us = 1x
            # else:
            #     s_us = GEN_STRIDE_UPSCALING

            #if mode == 'specgram':
            if False:
                y = y
                x = x
                samps_post_ds = y * x * n
                self.conv_layers.append(nn.Conv2d(c, n, kernel_size=1, bias=False, groups=1))
                self.conv_layers.append(nn.BatchNorm2d(n))
                self.conv_layers.append(nn.LeakyReLU(0.1, False))
                print("Created Upscaling layer with c={0} n={1} s_us={2} k_us={3}".format(c, n, s_us, k_us))
            else:
                orig_shape = dummy.shape
                if dummy.numel() // dummy.shape[0] <= GEN_MAX_LIN_FEATURES:
                    dummy = dummy.contiguous().view(BATCH_SIZE, -1)
                    self.conv_layers.append(nn.LazyLinear(dummy.shape[-1], bias=self.use_bias))
                    dummy = self.conv_layers[-1](dummy)
                    dummy = dummy.view(dummy.shape[0], orig_shape[1], -1)
                    v_cprint("Created Linear layer with", samps_post_ds, "output samples.")
                
                self.conv_layers.append(us(n, c, k_us, s_us, groups=n, bias=self.use_bias))
                #he_init(self.conv_layers[-1])
                #self.conv_layers[-1].weight.data.copy_(he_init(self.conv_layers[-1].weight))
                #icnr2d(self.conv_layers[-1].weight, he_init)
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Upscaling layer.", samps_post_ds, "<", dummy.numel() // BATCH_SIZE)
                samps_post_us = dummy.numel() // dummy.shape[0]

                self.conv_layers.append(norm(c))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Normalization layer with", c, "channels.")

                self.conv_layers.append(nn.Sigmoid())
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Activation layer.")
                
                self.conv_layers.append(ds(c, n, 1, 2, groups=n, bias=self.use_bias))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Grouped Convolution layer with", n, "groups.") 
                self.conv_layers.append(ds(n, n, 2, 1, groups=n, bias=self.use_bias))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Grouped Convolution layer with", n, "groups.")  
                self.conv_layers.append(ds(n, c, 1, 1, groups=n, bias=self.use_bias))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Grouped Convolution layer with", n, "groups.")            
                
                samps_post_us = dummy.numel() // dummy.shape[0]

                self.conv_layers.append(norm(c))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Normalization layer with", c, "channels.")

                
                # self.conv_layers.append(nn.ReplicationPad2d((self.blur_amount,0,self.blur_amount,0)))
                # dummy = self.conv_layers[-1](dummy)
                # v_cprint("Created Padding layer.")

                # self.conv_layers.append(nn.AvgPool2d(3, 1))
                # dummy = self.conv_layers[-1](dummy)
                # v_cprint("Created Adaptive Pooling layer.")
                


                if samps_post_us >= self.total_samp_out * 2:
                    #factor = max(min(dummy.shape[-1], dummy.shape[-2]) - 1, 1)
                    #factor = max(dummy.shape[-1] - 1, 1)
                    self.conv_layers.append(nn.Conv1d(c, c, 1, 2))
                    v_cprint("Created Conv1d layer with stride 2.")
                    dummy = self.conv_layers[-1](dummy)
                    

                # orig_shape = dummy.shape
                # dummy = dummy.view(dummy.shape[0], dummy.shape[1]*4, -1)
                # self.conv_layers.append(nn.LSTM(dummy.shape[-1], dummy.shape[-1], 1, bias=False, batch_first=True))
                # v_cprint("Created RNN layer with", self.conv_layers[-1].input_size, "input features and", self.conv_layers[-1].hidden_size, "hidden features.")
                # dummy, state = self.conv_layers[-1](dummy)
                # dummy = dummy.contiguous().view(orig_shape)
                # self.rnn_states[rnn_index] = (state[0].to(self.device).detach(), state[1].to(self.device).detach())
                # rnn_index += 1
                #self.rnn_states[len(self.conv_layers)] = dummy_state.to(self.device)
                
                self.conv_layers.append(shuf(n//2))
                dummy = self.conv_layers[-1](dummy)
                #self.conv_layers.append(TA)
                v_cprint("Created Shuffle layer with scaling factor {}.".format(n//2))

                self.conv_layers.append(ds(dummy.shape[1], n, 1, 1, bias=False))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Downscaling layer from {} to {} channels.".format(self.conv_layers[-1].in_channels, n))


                samps_post_ds = dummy.shape[-1] * self.n_channels
                if samps_post_ds < self.total_samp_out:
                    self.conv_layers.append(nn.ReLU())
                    dummy = self.conv_layers[-1](dummy)
                    v_cprint("Created Activation layer.")
                
                
                # if ds.__name__.find('2d') != -1:
                #     # samps_post_ds = int((np.sqrt(samps_post_us) - (k_ds-1) - 1) // s_ds + 1)**2
                #     # samps_post_ds = min(samps_post_ds, self.total_samp_out)
                #     x = (x - (1    - 1) - 1) // 1    + 1
                #     y = (y - (k_ds - 1) - 1) // s_ds + 1
                #     x = min(x, self.n_fft)
                #     y = min(y, self.total_samp_out)
                #     samps_post_ds = x*y
                #     self.conv_layers.append(ds(c, c, (1, k_ds), (1, s_ds), groups=c, bias=self.use_bias))
                #     self.conv_layers.append(pool((x, y)))
                # elif ds.__name__.find('1d') != -1:
                #     y = int((samps_post_us - (k_ds-1) - 1) // s_ds + 1)
                #     y = min(y, self.total_samp_out)
                #     samps_post_ds = y
                #     self.conv_layers.append(ds(c, c, k_ds, s_ds, groups=c, bias=self.use_bias))
                #     self.conv_layers.append(pool(samps_post_ds))
                # v_cprint("Created Downscaling layer.", samps_post_us, ">", samps_post_ds)
                
                # if y*x < self.total_samp_out*self.n_fft:
                #     self.conv_layers.append(nn.ReLU(inplace=False))
                #     v_cprint("Created Activation layer.")

                # self.conv_layers.append(norm(c))
                # v_cprint("Created Normalization layer with", c, "channels.")

            
            if self.n_sets == 1:
                self.n_layers_per_set = len(self.conv_layers)
            
            
            if samps_post_ds < 1:
                raise RuntimeError("Generator reached 0 samples after this many sets:", self.n_sets)
            
            v_cprint("Samples so far:", samps_post_ds)
            v_cprint("Still need:", self.total_samp_out - samps_post_ds)
        """
        
        print("Created", self.n_sets, "sets of generator processing layers.")
        
        # if self.is_2d:
        #     dummy = tf.reshape(dummy, [self.n_channels, self.n_fft-1, -1])
        # else:
        #     dummy = tf.reshape(dummy, [-1, 1])

        # self.net += [SubpixelUpscaling1D()]
        # dummy = self.net[-1](dummy)
        
        v_cprint("Created final layers.")
        v_cprint("Final shape:", list(dummy.shape))

        # if self.is_2d:
        #     dummy = torch.cat((dummy, torch.zeros((dummy.shape[0], dummy.shape[1], 1, dummy.shape[3])).to(dummy.device)), -2)
        self.net = keras.Sequential(self.net)
        for layer in self.net.layers:
            if not layer.trainable:
                layer.trainable = True
        # self.net.call(generate_input_noise())
        # self.net.summary()
        print()

    def sanity_check(self, data):
        # if torch.isnan(data).any():
        #     raise RuntimeError("Data is NaN!")
        pass

    #@tf.function
    def run_conv_net(self, inp, mode, training):
        #print(inp.min().item(), inp.max().item(), inp.mean().item())
        data = normalize_negone_one(tf.cast(inp, tf.float32))
        #print(data.min().item(), data.max().item(), data.mean().item())
        #vv_cprint("|}} Initial data.shape:", data.shape)

        self.sanity_check(data)

        post_final = self.net(data, training=training)

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
    
    def gen_fn(self, inputs, training):
        return self.run_conv_net(inputs, 'specgram', training)[:, :TOTAL_SAMPLES_OUT]
        