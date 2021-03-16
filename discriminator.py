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

class TADiscriminator(keras.Model):
    def __init__(self, *args, **kwargs):
        super(TADiscriminator, self).__init__(*args, **kwargs)

        self.ksz = DIS_KERNEL_SIZE
        self.stride = DIS_STRIDE
        if DIS_MODE in [0, 3]:
            self.ndf = 2
            self.samps = TOTAL_SAMPLES_OUT * 2
            x = self.samps
        elif DIS_MODE == 1:
            self.ndf = TOTAL_SAMPLES_OUT
        elif DIS_MODE == 2:
            self.ndf = 1
            x = DIS_N_MELS
            y = librosa.samples_to_frames(TOTAL_SAMPLES_OUT, DIS_HOP_LEN, DIS_N_FFT)
            self.samps = x*y
        else:
            raise ValueError("Invalid discriminator mode!")
        

        self.net = []
        #self.net.append(nn.Identity())
        i = 0
        c = self.ndf * 2
        n = self.ndf
        k1 = self.ksz
        s1 = self.stride
        k2 = self.ksz
        s2 = self.stride
        while self.samps > 256:
            c = min(DIS_MAX_CHANNELS, int(self.ndf * 2**i))
            n = min(DIS_MAX_CHANNELS, int(self.ndf * 2**(i+1)))
            
            # if y <= self.ksz:
            #     k2 = 1
            #     s2 = 1
            
            if x <= self.ksz:
                s1 = 1
                k1 = 1
            
            i += 1
            #sqrt_samps = int(np.sqrt(self.samps))
            x = (x - (k1 - 1) - 1) // s1 + 1
            #y = (y - (k2 - 1) - 1) // s2 + 1
            self.samps = x
            self.net.append(layers.Dropout(DIS_DROPOUT))
            self.net.append(layers.Conv1D(n, k1, s1, use_bias=False))
            self.net.append(layers.BatchNormalization())
            self.net.append(layers.LeakyReLU(0.2))
            #v_cprint("Created Conv2d layer #{6} with c={0} n={1} k=({2}, {3}) s=({4}, {5})\tsamps={7}".format(c, n, k1, s1, i, self.samps))
        print("Created", i, "sets of Discriminator layers.")
        x = (x - (k1 - 1) - 1) // s1 + 1
        #y = (y - (k2 - 1) - 1) // s2 + 1
        #self.samps = x*y
        #self.net.append(layers.Conv1D(1, 1, 1, use_bias=False))
        self.net.append(layers.Flatten())
        self.net.append(layers.Dense(128, use_bias=False))
        self.net.append(layers.Dense(64, use_bias=False))
        self.net.append(layers.Dense(32, use_bias=False))
        self.net.append(layers.Dense(16, use_bias=False))
        self.net.append(layers.Dense(8, use_bias=False))
        self.net.append(layers.Dense(1, use_bias=False, activation='sigmoid'))
    
    def call(self, inp):
        if DIS_MODE == 0:
            raise NotImplementedError
        elif DIS_MODE == 1:
            raise NotImplementedError("NYI")
            # fft1 = ag.Variable(torch.unsqueeze(
            #     torch.fft.fft(inp.to(torch.float), norm='forward'), -1), requires_grad=True)
            # mag = ag.Variable(torch.square(torch.real(fft1)) + torch.square(torch.imag(fft1)), requires_grad=True)
            # pha = ag.Variable(torch.atan2(torch.real(fft1), torch.imag(fft1)), requires_grad=True)
            # actual_input = ag.Variable(torch.stack((mag, pha)).permute(0,1,2), requires_grad=True)
        elif DIS_MODE == 2:
            raise NotImplementedError
            # # what an ugly line of code
            # if N_GEN_MEL_CHANNELS in inp.shape or DIS_N_MELS in inp.shape:
            #     actual_input = inp.to(torch.float).unsqueeze(0)
            # else:
            #     melspec = AudioToMelWithGradients.apply(inp.to(torch.float), DIS_N_FFT, DIS_N_MELS, DIS_HOP_LEN).requires_grad_(True)
            #     if torch.is_grad_enabled():
            #         melspec.retain_grad()
            #     actual_input = melspec.unsqueeze(1).requires_grad_(True)
            #     if torch.is_grad_enabled():
            #         actual_input.retain_grad()
        elif DIS_MODE == 3:
            actual_input = tf.reshape(audio_to_specgram(tf.reshape(inp, [inp.shape[0], -1])), [inp.shape[0], self.ndf, -1])[..., :TOTAL_SAMPLES_OUT]

        verdicts = actual_input
        for i, layer in enumerate(self.net):
            verdicts = layer(verdicts)
        
        return tf.clip_by_value(tf.squeeze(verdicts), clip_value_min=0.001, clip_value_max=0.999)
