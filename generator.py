import six
import os
import time
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchaudio

#from online_norm_pytorch import OnlineNorm1d, OnlineNorm2d

from helper import *
from hilbert import *
from global_constants import *

class TAInstParamGenerator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if GEN_MODE not in [5]:
            raise ValueError("Created wrong generator class!")
        
        self.loss = nn.BCELoss()

        self.n_layers = GEN_MIN_LAYERS
        self.ksz = GEN_KERNEL_SIZE_DOWNSCALING
        self.ndf = N_PARAMS * 2 * TOTAL_PARAM_UPDATES # ensure we never run out of params

        self.layers = [ct(TOTAL_SAMPLES_IN, self.ndf, self.ksz)]
        i = 0
        for _ in range(self.n_layers):
            c = 2**i
            n = 2**(i+1)
            #self.layers.append(nn.Conv1d(self.ndf*c, self.ndf*n, self.ksz, 2, 1, bias=False))
            #self.layers.append(nn.BatchNorm1d(self.ndf*n))
            self.layers.append(nn.Flatten())
            self.layers.append(nn.Linear(self.ndf, self.ndf))
            #self.layers.append(nn.LeakyReLU(0.2, inplace=True))
            i += 1

        #self.layers.append(nn.Conv1d(self.ndf, self.ndf//2, self.ksz, 1, 0, bias=False))
        #self.layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*self.layers)
    
    def forward(self, inp):
        a = self.layers.forward(inp).flatten()
        return a/torch.max(torch.abs(a))

def he_initializer_scale(shape, slope=1.0):
    return np.sqrt(2. / ((1. + slope**2) * np.prod(shape[:-1])))

def pixel_norm(data, epsilon=1.0e-8, dim=3):
    return data * torch.rsqrt(torch.mean(torch.square(data), dim=dim, keepdim=True) + epsilon)

class TADownscalingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_shape, strides, he_slope):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        if type(kernel_shape) is tuple:
            self.dim = len(kernel_shape)
            weights = torch.Tensor(in_channels, out_channels, *kernel_shape)
        else:
            self.dim = 1
            weights = torch.Tensor(in_channels, out_channels, kernel_shape)
        
        bias = torch.zeros(out_channels)
        self.bias = nn.Parameter(bias)
        
        weights_shape = weights.shape

        self.strides = strides

        kernel_scale = he_initializer_scale(weights_shape, he_slope)
        init_scale, post_scale = 1.0, kernel_scale
        nn.init.normal_(weights, std=init_scale)
        weights_scaled = post_scale * weights + self.bias
        weights = pixel_norm(F.relu(weights_scaled), dim=self.dim+1)
        self.weight = nn.Parameter(weights)
    
    def forward(self, inp):
        if self.dim > 1:
            return F.conv2d(inp, self.weight, self.bias, self.strides,
                            0, 1, 1)
        else:
            return F.conv1d(inp, self.weight, self.bias, self.strides,
                            0, 1, 1)

class TARNNUpscalingLayer(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.layer = nn.LSTM(in_features, hidden_features, 1, batch_first=True)
        self.state = None
    def forward(self, inp):
        out, self.state = self.layer.forward(inp, self.state)
        return out

def icnr1d(x, scale=2, init=nn.init.kaiming_normal_):
    """ICNR init of `x`, with `scale` and `init` function.
    Source: https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    """
    ni,nf,h = x.shape
    ni2 = int(ni/scale)
    k = init(torch.zeros([ni2,nf,h])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale)
    k = k.contiguous().view([nf,ni,h]).transpose(0, 1)
    x.data.copy_(k)

def icnr2d(x, scale=2, init=nn.init.kaiming_normal_):
    """ICNR init of `x`, with `scale` and `init` function.
    Source: https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)

def he_init(x, he_slope=0.1):
    weights = x.weight.clone()
    weights_shape = weights.shape

    kernel_scale = he_initializer_scale(weights_shape, he_slope)
    init_scale, post_scale = 1.0, kernel_scale
    nn.init.normal_(weights, std=init_scale)
    weights_scaled = post_scale * weights
    weights = pixel_norm(F.relu(weights_scaled))
    x.weight.data.copy_(weights)

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output

    Taken from https://github.com/serkansulun/pytorch-pixelshuffle1d
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output

    Taken from https://github.com/serkansulun/pytorch-pixelshuffle1d
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x

class TASimpleReshape1d(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    
    def forward(self, x):
        return x.view(x.shape[0], x.shape[1] // self.scale, -1)

class TASmartShuffle1d(nn.Module):
    def __init__(self, scale, weight_size = GEN_MAX_CHANNELS):
        super().__init__()
        assert(type(scale) is int and scale > 0)
        self.scale = scale
        weights = torch.Tensor(weight_size, 2) # just some high number, if you can precalculate this then just put it in here!
        
        weights_shape = weights.shape

        kernel_scale = he_initializer_scale(weights_shape, 0.1)
        init_scale, post_scale = 1.0, kernel_scale
        nn.init.normal_(weights, std=init_scale)
        #weights_scaled = post_scale * weights
        #weights = pixel_norm(weights_scaled, dim=0)
        self.weight = nn.Parameter(weights)
    
    def forward(self, x):
        y = x.permute(0, 2, 1)
        out_timesteps = y.shape[-1] * self.scale
        out_channels = y.shape[1] // self.scale
        total = y.shape[1] * y.shape[-1] - 1
        # x = torch.Tensor([
        #     [0, 1, 2,],
        #     [3, 4, 5,],
        #     [6, 7, 8,],
        #     [9, 10, 11,],
        # ])
        
        z = x.view(x.shape[0], -1)
        out = x.view(x.shape[0], out_channels, out_timesteps)
        #mappings_chans = torch.linspace(0, self.out_timesteps, self.in_channels)
        for k in range(x.shape[0]):
            for i in range(out_channels):
                idx1 = int(self.weight[i][0] * total**2) % out_channels
                idx2 = int(self.weight[i][1] * total**2) % out_channels
                out[k][idx1] = z[k][idx2]
        return out


def ta_upscaling_layer(in_channels, out_channels, in_timesteps, scale, ksz):
    conv = nn.Conv1d(in_channels, out_channels*scale, kernel_size=ksz, bias=False, groups=1)
    norm = nn.BatchNorm1d(out_channels*scale)
    #nn.utils.weight_norm(self.conv)
    icnr1d(conv.weight)
    #shuf = TAShuffle1d(scale)
    shuf = TASmartShuffle1d(scale)
    pad = nn.ReplicationPad1d((1,0))
    blur = nn.AvgPool1d(2, stride=1)
    relu = nn.LeakyReLU(0.1, False)
    return conv, norm, relu, shuf, pad, blur

class TAUpscalingLayer2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*(scale**2), kernel_size=1, bias=False, groups=in_channels)
        self.norm = nn.BatchNorm2d(out_channels*(scale**2))
        #nn.utils.weight_norm(self.conv)
        #icnr2d(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(True)
        self.net = nn.Sequential(
            self.conv,
            self.norm,
            self.relu,
            self.shuf,
            self.pad,
            self.blur,
        )
    
    def forward(self, inp):
        return self.net(inp)

def random_phase_shuffle(inp, scale):
    out = inp.clone()
    second_half = inp.shape[1] // 2
    noise = torch.randn((inp.shape[0], inp.shape[1]-second_half, inp.shape[2], inp.shape[3])).to(inp.device) \
        * scale * inp[:,:-second_half].abs().max()
    out[:,:-second_half] = inp[:,:-second_half] + noise
    return out

class TAGenerator(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if GEN_MODE == 0:
            #TODO: implement RNN/Hilbert mode
            raise NotImplementedError("TODO: implement RNN/Hilbert mode")
        elif GEN_MODE == 1:
            #TODO: port RNN/Audio code to pytorch
            #raise NotImplementedError("TODO: implement RNNConv/Audio mode")
            self.create_conv_net(mode='rnnaudio')
        elif GEN_MODE == 2:
            self.create_conv_net(mode='hilbert')
        elif GEN_MODE == 3:
            self.create_conv_net(mode='audio')
        elif GEN_MODE == 4:
            self.create_conv_net(mode='mel')
        elif GEN_MODE == 5:
            self.create_conv_net(mode='stft')
        elif GEN_MODE == 6:
            self.create_conv_net(mode='specgram')
        else:
            # self.total_samp_out = TOTAL_PARAM_UPDATES * N_PARAMS
            # self.desired_process_units = TOTAL_SAMPLES_IN
            # self.create_dense_net(hilb_mode=False)
            raise ValueError("Created wrong generator class!")
    
    def create_conv_net(self, mode='audio'):
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
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            norm = nn.BatchNorm2d
            pool = nn.AdaptiveAvgPool2d
            shuf = nn.PixelShuffle
        elif mode == 'mel':
            self.n_channels = 1
            self.n_fft = N_GEN_MEL_CHANNELS
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_fft
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_INITIAL_LIN_SCALE * self.n_fft
            self.total_samp_out = GEN_N_FRAMES
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            norm = nn.BatchNorm2d
            pool = nn.AdaptiveAvgPool2d
            shuf = nn.PixelShuffle
        elif mode == 'stft':
            self.n_channels = 2
            self.n_fft = N_GEN_FFT // 2
            self.linear_units_in = TOTAL_SAMPLES_IN * 2 * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_INITIAL_LIN_SCALE * self.n_channels
            self.initial_n_fft = self.linear_units_out // 2
            self.total_samp_out = int(GEN_N_FRAMES * 1.1)
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            norm = nn.BatchNorm2d
            pool = nn.AdaptiveAvgPool2d
            shuf = nn.PixelShuffle
        elif mode == 'specgram':
            self.n_channels = 2
            self.n_fft = 1
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_INITIAL_LIN_SCALE * self.n_channels
            self.total_samp_out = TOTAL_SAMPLES_OUT * self.n_channels
            self.initial_n_fft = 1
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            norm = nn.BatchNorm2d
            pool = nn.AdaptiveAvgPool2d
            shuf = nn.PixelShuffle
        lin = nn.Linear
        rs = torchaudio.transforms.Resample

        self.padding_mode = 'reflect'
        self.use_bias = True
        self.sr = SAMPLE_RATE

        self.rnn_states = {}

        v_cprint("*-"*39 + "*")
        v_cprint("Target length is", self.total_samp_out, "samples.")

        dummy = torch.randn(BATCH_SIZE, self.linear_units_in).to(self.device)

        self.initial_layers = [
            nn.Linear(self.linear_units_in, self.linear_units_out, bias=self.use_bias),
        ]
        self.initial_layers = nn.Sequential(*self.initial_layers)
        dummy = self.initial_layers(dummy)
        v_cprint("Created Activation layer.")
        v_cprint("Created Linear layer.", self.linear_units_in, ">", self.linear_units_out)
        v_cprint("Created Activation layer.")
        
        # if mode == 'mel':
        #     v_cprint("Created Mel Spectrogram layer.")
        #     self.initial_layers.append(torchaudio.transforms.MelSpectrogram(self.sr, TOTAL_SAMPLES_IN, n_mels=self.n_channels))
        
        self.n_sets = 0
        prob = 0.01
        s_us = 1
        k_us = 1
        s_ds = 1
        k_ds = 1
        samps_post_ds = y
        c = self.n_channels
        n = self.n_channels*2
        dummy = dummy.view(BATCH_SIZE, 1, 2, -1)
        rnn_index = 0
        while samps_post_ds < self.total_samp_out*self.n_fft or self.n_sets < GEN_MIN_LAYERS:
            self.n_sets += 1
            #k_us = GEN_KERNEL_SIZE_UPSCALING * self.n_sets + 1
            #k_ds = GEN_KERNEL_SIZE_DOWNSCALING * max(self.n_sets // 25, 1)
            #n = min(GEN_MAX_CHANNELS, (2**(self.n_sets)))
            n = GEN_STRIDE_UPSCALING
            c = min(n**2, GEN_MAX_CHANNELS)
            v_cprint("="*80)

            # if samps_post_ds >= self.total_samp_out:
            #     s_us = 1x
            # else:
            #     s_us = GEN_STRIDE_UPSCALING

            #if mode == 'specgram':
            if False:
                #k_us = (n + 420691337) % (c-1) + 1
                y = y
                x = x
                samps_post_ds = y * x * n
                #self.conv_layers.append(nn.Sequential(*ta_upscaling_layer(c, n, y//s_us, s_us,k_us)))
                
                #icnr1d(conv.weight)
                self.conv_layers.append(nn.Conv2d(c, n, kernel_size=1, bias=False, groups=1))
                self.conv_layers.append(nn.BatchNorm2d(n))
                #nn.utils.weight_norm(self.conv)
                #self.conv_layers.append(nn.PixelShuffle(s_us))
                #shuf = TAShuffle1d(scale)
                #self.conv_layers.append(TASmartShuffle1d(s_us))
                #self.conv_layers.append(nn.ReplicationPad2d((1,0,1,0)))
                #self.conv_layers.append(nn.AvgPool2d(2, stride=1))
                self.conv_layers.append(nn.LeakyReLU(0.1, False))
                print("Created Upscaling layer with c={0} n={1} s_us={2} k_us={3}".format(c, n, s_us, k_us))
                #self.conv_layers.append(ds(c, c, 1, s_ds))
                #self.conv_layers.append(nn.ReLU(False))
                #self.conv_layers.append(nn.BatchNorm1d(c))
            #elif mode == 'stft':
            #    y = y * s_us
            #    samps_post_ds = y
            #    self.conv_layers.append(TAUpscalingLayer2d(c, n, s_us))
            else:
                #self.conv_layers.append(nn.Dropout(prob, False))
                #dummy = self.conv_layers[-1](dummy)
                #v_cprint("Created Dropout layer with", prob, "probability.")
                
                orig_shape = dummy.shape
                if samps_post_ds <= GEN_MAX_LIN_FEATURES:
                    self.conv_layers.append(nn.LazyLinear(samps_post_ds, bias=self.use_bias))
                    dummy = dummy.contiguous().view(BATCH_SIZE, -1)
                    dummy = self.conv_layers[-1](dummy)
                    dummy = dummy.view(dummy.shape[0], orig_shape[1], 1, -1)
                    v_cprint("Created Linear layer with", samps_post_ds, "output samples.")
                self.conv_layers.append(ds(1, c, k_us, s_us, groups=1, bias=self.use_bias))
                he_init(self.conv_layers[-1])
                dummy = self.conv_layers[-1](dummy)

                v_cprint("Created Upscaling layer.", samps_post_ds, "<", dummy.numel() // BATCH_SIZE)
                samps_post_us = dummy.numel() // dummy.shape[0]

                self.conv_layers.append(norm(c))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Normalization layer with", c, "channels.")

                self.conv_layers.append(nn.ReLU(inplace=False))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Activation layer.")
                
                self.conv_layers.append(ds(c, c*2, 1, 1, groups=c, bias=self.use_bias))
                he_init(self.conv_layers[-1])
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Grouped Convolution layer with", c, "groups.")
                self.conv_layers.append(ds(c*2, c*4, 1, 1, groups=c*2, bias=self.use_bias))
                he_init(self.conv_layers[-1])
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Grouped Convolution layer with", c*2, "groups.")  
                self.conv_layers.append(ds(c*4, c*2, 1, 1, groups=c*2, bias=self.use_bias))
                he_init(self.conv_layers[-1])
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Grouped Convolution layer with", c*2, "groups.")  
                self.conv_layers.append(ds(c*2, c, 1, 1, groups=c, bias=self.use_bias))
                he_init(self.conv_layers[-1])
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Grouped Convolution layer with", c, "groups.")            
                samps_post_us = dummy.numel() // dummy.shape[0]

                self.conv_layers.append(norm(c))
                dummy = self.conv_layers[-1](dummy)
                v_cprint("Created Normalization layer with", c, "channels.")

                
                if samps_post_us >= self.total_samp_out:
                    factor = max(min(dummy.shape[-1], dummy.shape[-2]) - 1, 1)
                    self.conv_layers.append(nn.AvgPool2d(factor, stride=1))
                    v_cprint("Created Pooling layer with factor", factor)
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
                
                if int(np.sqrt(c)) > 1:
                    self.conv_layers.append(shuf(int(np.sqrt(c))))
                    dummy = self.conv_layers[-1](dummy)
                    #self.conv_layers.append(TA)
                    v_cprint("Created Shuffle layer with scaling factor", int(np.sqrt(c)), ".")

                if dummy.numel() // BATCH_SIZE < self.total_samp_out:
                    self.conv_layers.append(nn.ReLU(inplace=False))
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
            
            samps_post_ds = dummy.numel() // BATCH_SIZE
            if samps_post_ds < 1:
                raise RuntimeError("Generator reached 0 samples after this many sets:", self.n_sets)
            
            v_cprint("Samples so far:", samps_post_ds)
            v_cprint("Still need:", self.total_samp_out - samps_post_ds)
        
        
        v_cprint("="*80)
        print("Created", self.n_sets, "sets of", self.n_layers_per_set, " generator processing layers.")

        #v_cprint("Created Linear layer.", self.total_samp_out, ">", self.n_channels * self.n_bins**2)
        self.lin_layers = [
            #lin(self.total_samp_out, self.n_channels * self.n_bins**2)
        ]
        if samps_post_ds > self.total_samp_out*self.n_fft:
            self.final_layers = [
                # nn.Flatten(),
                #ds(c, self.n_channels, 1, 1, groups=self.n_channels, bias=False),
                
                #nn.AvgPool2d(samps_post_ds // self.total_samp_out, 1),
                #nn.PixelShuffle(2),
                #nn.Tanh(),
                nn.Identity(),
                # nn.Flatten(),
            ]
        else:
            self.final_layers = [nn.Identity()]
        v_cprint("Created final layers.")
        
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.final_layers = nn.Sequential(*self.final_layers)
        dummy = self.final_layers(dummy)
        v_cprint("Final shape:", list(dummy.shape))
        print()

    def sanity_check(self, data):
        if torch.isnan(data).any():
            raise RuntimeError("Data is NaN!")
        pass

    def run_conv_net(self, inp, mode):
        data = normalize_data(inp.float()).to(self.device)
        batch_sz = data.shape[0]
        if self.training:
            data.retain_grad()
        vv_cprint("|}} Initial data.shape:", data.shape)
        if mode == 'mel':
            data = self.initial_layers(data.view(batch_sz, self.linear_units_in))
            data = data.reshape((batch_sz, 1, self.n_fft, -1))
        elif mode == 'stft':
            post_init_audio = self.initial_layers(data.view(batch_sz, self.linear_units_in))
            post_init = audio_to_stftgram(post_init_audio.view(batch_sz, -1), self.initial_n_fft, self.initial_n_fft//4)#.view(BATCH_SIZE, self.n_channels, self.initial_n_fft, -1)
        elif mode == 'specgram':
            post_init_audio = self.initial_layers(data.view(batch_sz, self.linear_units_in).to(self.device))
            #post_init = audio_to_specgram(post_init_audio.view(batch_sz, -1)).view(batch_sz, self.n_channels, 2, -1)
            post_init = post_init_audio.view(batch_sz, 1, 2, -1)
        else:
            data = data.view(batch_sz, self.linear_units_in)
            data = self.initial_layers(data)
            data = data.view(batch_sz, self.n_channels, 1, -1)

        if self.training:
            post_init.retain_grad()
        
        self.sanity_check(post_init)

        def fix_size(x, dim, axis=-1):
            quot = x.shape[-1] / dim
            diff = abs(int(quot) - quot)
            k = 0
            while diff > 0.0:
                k += 1
                quot = (x.shape[-1]-k) / dim
                if k == x.shape[-1]:
                    raise RuntimeError("Could not truncate signal to fit dimensions!")
                diff = abs(int(quot) - quot)
            if k > 0:
                x = x[..., :-k]
            return x
        
        def check_and_fix_size(y, dim, axis=-1):
            len_shape = len(y.shape)
            total = y.shape[axis]
            diff = dim - total
            if diff > 0:
                x = fix_size(y, dim)
                if axis == -1:
                    if len_shape == 2:
                        return x.reshape(x.shape[0], -1)
                    elif len_shape == 3:
                        return x.reshape(x.shape[0], x.shape[1], -1)
                    elif len_shape == 4:
                        return x.reshape(x.shape[0], x.shape[1], x.shape[2], -1)
                    else:
                        raise ValueError("check_and_fix_size() # of dims must be < 5 and > 1")
                elif axis == -2:
                    if len_shape == 2:
                        return x.reshape(-1, x.shape[1])
                    elif len_shape == 3:
                        return x.reshape(x.shape[0], -1, x.shape[2])
                    elif len_shape == 4:
                        return x.reshape(x.shape[0], x.shape[1], -1, x.shape[3])
                    else:
                        raise ValueError("check_and_fix_size() # of dims must be < 5 and > 1")
                elif axis == -3:
                    if len_shape == 3:
                        return x.reshape(-1, x.shape[1], x.shape[2])
                    elif len_shape == 4:
                        return x.reshape(x.shape[0], -1, x.shape[2], x.shape[3])
                    else:
                        raise ValueError("check_and_fix_size() # of dims must be < 5 and > 1")
                else:
                    raise ValueError("check_and_fix_size() axis must be < 0 and > -4")
            elif diff < 0:
                if axis == -1:
                    return y[..., :diff]
                elif axis == -2:
                    return y[..., :diff, :]
                elif axis == -3:
                    return y[..., :diff, :, :]
                else:
                    raise ValueError("check_and_fix_size() axis must be < 0 and > -4")
            else:
                return y

        #vv_cprint("|}} Initial layers done.")

        #post_conv = self.conv_layers(post_init).requires_grad_(True)
        post_conv = post_init
        #timestamp1 = time.time()
        #j = 0
        if self.training:
            post_conv.retain_grad()
        rnn_index = 0
        for i, layer in enumerate(self.conv_layers):
            timestamp2 = time.time()
            if layer.__class__.__name__.find('Linear') != -1:
                orig_shape = post_conv.shape
                post_conv = post_conv.view(batch_sz, -1)
            if layer.__class__.__name__ in ['RNN', 'LSTM', 'GRU'] != -1:
                orig_shape = post_conv.shape
                post_conv = post_conv.view(post_conv.shape[0], post_conv.shape[1]*4, -1)
                post_conv, state = layer(post_conv, (self.rnn_states[rnn_index][0].to(post_conv.device), self.rnn_states[rnn_index][1].to(post_conv.device)))
                self.rnn_states[rnn_index] = (state[0].detach(), state[1].detach())
                post_conv = post_conv.contiguous().view(orig_shape)
                rnn_index += 1
            else:
                post_conv = layer(post_conv)
            if layer.__class__.__name__.find('Linear') != -1:
                post_conv = post_conv.view(post_conv.shape[0], orig_shape[1], 1, -1)
                #print(j, ":", layer.__class__.__name__, "done in", round(time.time()-timestamp2, 3), "seconds.")
            if layer.__class__.__name__.find('Conv') != -1 and self.training:
                post_conv = random_phase_shuffle(post_conv, GEN_PHASE_SHUFFLE)
            self.sanity_check(post_conv)
        

        # for i in range(self.n_sets):
        #     for j in range(self.n_layers_per_set):
        #         t = self.conv_layers[i*self.n_layers_per_set+j].__class__.__name__
        #         # if t.find('Conv') != -1 and self.training:
        #         #     for chan in range(data.shape[1]):
        #         #         data[:, chan] = F.normalize(data[:, chan], dim=-1)
        #         data = self.conv_layers[i*self.n_layers_per_set+j](data)
        #         #self.sanity_check(data)
        #         if mode not in ['specgram', 'stft']:
        #             data = check_and_fix_size(data, self.n_fft, -2)
        #             data = data.view(BATCH_SIZE, self.n_channels, self.n_fft, -1)

        #post_conv = torch.nan_to_num(post_conv, EPSILON, EPSILON, EPSILON)
        

        #vv_cprint("|}} Convolution layers done in", round(time.time()-timestamp1, 3), "seconds.")

        post_final = self.final_layers(post_conv)
        self.sanity_check(post_final)
        if self.training:
            post_final.retain_grad()
        self.sanity_check(post_final)
        if mode == 'mel' and DIS_MODE == 2:
            ret = check_and_fix_size(data.squeeze(), self.n_fft, -2)
        elif mode == 'mel':
            invmel_inp = check_and_fix_size(data, self.n_fft, -2)
            stft = MelToSTFTWithGradients.apply(invmel_inp, self.n_fft)
            data = stft_to_audio(stft, GEN_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_PREVIEW)
        elif mode == 'stft':
            pre_stftgram = post_final.reshape(batch_sz, self.n_channels, self.n_fft, -1).requires_grad_(True)#[..., :self.total_samp_out]
            if self.training:
                pre_stftgram.retain_grad()
            stftgram = torch.stack((normalize_data(torch.sigmoid(pre_stftgram[:,0])), torch.tanh(pre_stftgram[:,1])), dim=1)
            if self.training:
                stftgram.retain_grad()
            # stftgram[:, 0, :, :] = F.normalize(stftgram[0, :, :], dim=-1)
            # stftgram[:, 1, :, :] = F.normalize(stftgram[1, :, :], dim=-1)
            ret = stftgram_to_audio(stftgram, GEN_HOP_LEN)
            self.sanity_check(ret)
            #self.sanity_check(data)
        elif mode == 'specgram':
            pre_specgram = post_final[..., :self.total_samp_out].contiguous().view(batch_sz, self.n_channels, -1)
            if self.training:
                pre_specgram.retain_grad()
            specgram = torch.stack((normalize_data(torch.sigmoid(pre_specgram[:,0])), torch.tanh(pre_specgram[:,1])), dim=1).requires_grad_(True)
            if self.training:
                specgram.retain_grad()
            self.sanity_check(specgram)
            #print(specgram[:,0].min(), specgram[:,0].max())
            #print(specgram[:,1].min(), specgram[:,1].max())
            ret = specgram_to_audio(specgram.squeeze()[..., :TOTAL_SAMPLES_OUT]).requires_grad_(True)
            #ret = pre_specgram.view(batch_sz, -1)[..., :TOTAL_SAMPLES_OUT].requires_grad_(True)
            if self.training:
                ret.retain_grad()
            self.sanity_check(ret)
        else:
            data = normalize_data(data.view(-1)[:TOTAL_SAMPLES_OUT])
            #self.sanity_check(data)
        vv_cprint("|}} Final layers done.")
        vv_cprint("|}} data.shape:", ret.shape)
        write_normalized_audio_to_disk(ret[0], "./test1.wav")
        return ret

    def gen_conv(self, inputs, mode):
        assert(GEN_MODE in (1, 3, 4, 5, 6))
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        post_net = self.run_conv_net(inputs, mode)
        vv_cprint("|} Whew... Made it out of the net alive!")
        if mode == 'mel' and DIS_MODE == 2:
            return post_net
        else:
            return post_net.squeeze()[..., :TOTAL_SAMPLES_OUT]

    def gen_fn(self, inputs):
        if GEN_MODE == 1:
            return self.gen_conv(inputs, 'rnnaudio')
        elif GEN_MODE == 3:
            return self.gen_conv(inputs, 'audio')
        elif GEN_MODE == 4:
            return self.gen_conv(inputs, 'mel')
        elif GEN_MODE == 5:
            return self.gen_conv(inputs, 'stft')
        elif GEN_MODE == 6:
            return self.gen_conv(inputs, 'specgram')
        else:
            return None
    
    def forward(self, inputs):
        return self.gen_fn(inputs)
    
    def training_step(self, batch, batch_idx):
        return self(batch)
        