import six
import os
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchaudio

from online_norm_pytorch import OnlineNorm1d, OnlineNorm2d

from helper import *
from hilbert import *
from global_constants import *

class TAInstParamGenerator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if GEN_MODE not in [5]:
            raise ValueError("Created wrong generator class!")
        
        self.loss = nn.BCELoss()

        self.n_layers = MIN_N_GEN_LAYERS
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

class TAUpscalingLayer(nn.Module):
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
            return F.conv_transpose2d(inp, self.weight, self.bias, self.strides,
                            0, 1, 1)
        else:
            return F.conv_transpose1d(inp, self.weight, self.bias, self.strides,
                            0, 1, 1)

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
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_SCALE_LIN * self.n_channels
            self.total_samp_out = TOTAL_SAMPLES_OUT
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            norm = OnlineNorm2d
            pool = nn.AdaptiveAvgPool2d
        elif mode == 'mel':
            self.n_channels = 1
            self.n_fft = N_GEN_MEL_CHANNELS
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_fft
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_SCALE_LIN * self.n_fft
            self.total_samp_out = GEN_N_FRAMES
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            norm = OnlineNorm2d
            pool = nn.AdaptiveAvgPool2d
        elif mode == 'stft':
            self.n_channels = 2
            self.n_fft = N_GEN_FFT // 2 + 1
            self.linear_units_in = TOTAL_SAMPLES_IN * 2 * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_SCALE_LIN * self.n_channels
            self.initial_n_fft = self.linear_units_out // 2
            self.total_samp_out = int(GEN_N_FRAMES * 1.5)
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            norm = OnlineNorm2d
            pool = nn.AdaptiveAvgPool2d
        elif mode == 'specgram':
            self.n_channels = 2
            self.n_fft = 1
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels * self.n_fft
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_SCALE_LIN * self.n_channels * self.n_fft
            self.total_samp_out = int(TOTAL_SAMPLES_OUT * 1) # estimating how much we're gonna truncate by
            us = nn.ConvTranspose1d
            ds = nn.Conv1d
            #norm = OnlineNorm1d
            pool = nn.AdaptiveAvgPool1d
        lin = nn.Linear
        rs = torchaudio.transforms.Resample

        self.padding_mode = 'reflect'
        self.use_bias = False
        self.sr = SAMPLE_RATE

        v_cprint("*-"*39 + "*")
        v_cprint("Target length is", self.total_samp_out, "samples.")

        
        #self.initial_layer = nn.ReLU()
        self.initial_layers = [
            nn.ReLU(True),
            nn.Linear(self.linear_units_in, self.linear_units_out, bias=self.use_bias),
            nn.ReLU(True),
        ]
        v_cprint("Created Activation layer.")
        v_cprint("Created Linear layer.", self.linear_units_in, ">", self.linear_units_out)
        v_cprint("Created Activation layer.")
        
        # if mode == 'mel':
        #     v_cprint("Created Mel Spectrogram layer.")
        #     self.initial_layers.append(torchaudio.transforms.MelSpectrogram(self.sr, TOTAL_SAMPLES_IN, n_mels=self.n_channels))
        
        self.n_sets = 0
        prob = 0.1
        s_us = GEN_STRIDE_UPSCALING
        k_us = GEN_KERNEL_SIZE_UPSCALING
        s_ds = GEN_STRIDE_DOWNSCALING
        k_ds = GEN_KERNEL_SIZE_DOWNSCALING
        x = self.n_fft
        y = self.linear_units_out // self.n_fft
        samps_post_ds = y
        while y*x < self.total_samp_out*self.n_fft or self.n_sets < MIN_N_GEN_LAYERS:
            self.n_sets += 1
            #n = int(self.n_channels * s_us**(2*self.i))
            #c = int(self.n_channels * s_us**(2*self.i+1))
            #k_us = GEN_KERNEL_SIZE_UPSCALING * self.n_sets + 1
            #k_ds = GEN_KERNEL_SIZE_DOWNSCALING * max(self.n_sets // 25, 1)
            n = self.n_channels
            c = self.n_channels
            v_cprint("="*80)

            # if samps_post_ds >= self.total_samp_out:
            #     s_us = 1
            # else:
            #     s_us = GEN_STRIDE_UPSCALING

            if us.__name__.find('2d') != -1:
                #samps_post_us = int(np.sqrt(samps_post_ds-1) * s_us + (k_us-1) + 1)**2
                #sqrt_samps_post_ds = int(np.sqrt(samps_post_ds))
                x = (x - 1) * 1    + (1 - 1) + 1
                y = (y - 1) * s_us + (k_us - 1) + 1
                samps_post_us = x*y
                self.conv_layers.append(us(c, n, (1, k_us), (1, s_us), groups=1, bias=self.use_bias))
            elif us.__name__.find('1d') != -1:
                y = (y - 1) * s_us + (k_us - 1) + 1
                samps_post_us = y
                self.conv_layers.append(us(c, n, k_us, s_us, groups=1, bias=self.use_bias))
            #self.conv_layers.append(rs(2**self.n_sets, s_us * 2**(self.n_sets+1)))
            v_cprint("Created Upscaling layer.", samps_post_ds, "<", samps_post_us)
            
            self.conv_layers.append(nn.LeakyReLU(0.1, False))
            v_cprint("Created Activation layer.")

            self.conv_layers.append(nn.Dropout(prob, False))
            v_cprint("Created Dropout layer with", prob, "probability.")
            
            if ds.__name__.find('2d') != -1:
                # samps_post_ds = int((np.sqrt(samps_post_us) - (k_ds-1) - 1) // s_ds + 1)**2
                # samps_post_ds = min(samps_post_ds, self.total_samp_out)
                x = (x - (1    - 1) - 1) // 1    + 1
                y = (y - (k_ds - 1) - 1) // s_ds + 1
                x = min(x, self.n_fft)
                y = min(y, self.total_samp_out)
                samps_post_ds = x*y
                self.conv_layers.append(ds(n, c, (1, k_ds), (1, s_ds), groups=1, bias=self.use_bias))
                self.conv_layers.append(pool((x, y)))
            elif ds.__name__.find('1d') != -1:
                y = int((samps_post_us - (k_ds-1) - 1) // s_ds + 1)
                y = min(y, self.total_samp_out)
                samps_post_ds = y
                self.conv_layers.append(ds(n, c, k_ds, s_ds, groups=1, bias=self.use_bias))
                self.conv_layers.append(pool((samps_post_ds)))
            v_cprint("Created Downscaling layer.", samps_post_us, ">", samps_post_ds)
            
            self.conv_layers.append(nn.LeakyReLU(0.1, False))
            v_cprint("Created Activation layer.")

            #self.conv_layers.append(norm(c))
            #v_cprint("Created Normalization layer with", c, "channels.")

            
            if self.n_sets == 1:
                self.n_layers_per_set = len(self.conv_layers)

            v_cprint("Samples so far:", samps_post_ds)
            v_cprint("Still need:", self.total_samp_out - samps_post_ds)
        
        
        v_cprint("="*80)
        v_cprint("Created", self.n_sets, "sets of", self.n_layers_per_set, "processing layers.")

        #v_cprint("Created Linear layer.", self.total_samp_out, ">", self.n_channels * self.n_bins**2)
        self.lin_layers = [
            #lin(self.total_samp_out, self.n_channels * self.n_bins**2)
        ]

        if pool.__name__.find('2d') != -1:
            self.final_layers = [
                # torchaudio.transforms.InverseMelScale(self.n_bins, self.n_channels, self.sr),
                # nn.Flatten(),
                pool((self.n_fft, self.total_samp_out)),
                #nn.Tanh(),
                #nn.Identity(),
                # #torchaudio.transforms.Resample(self.sr, SAMPLE_RATE),
                # nn.Flatten(),
            ]
        else:
            self.final_layers = [
                #pool(self.total_samp_out),
                nn.Identity(),
            ]
        v_cprint("Created final layers.")
        
        self.initial_layers = nn.Sequential(*self.initial_layers)
        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.final_layers = nn.Sequential(*self.final_layers)

    def sanity_check(self, data):
        if (not torch.isfinite(data.view(-1)[0])) or (torch.isnan(data.view(-1)[0])):
            raise RuntimeError("Data is NaN!")
        pass

    def run_conv_net(self, inp, mode):
        data = normalize_data(inp.float())
        if self.training:
            data.retain_grad()
        vv_cprint("|}} Initial data.shape:", data.shape)
        if mode == 'mel':
            # no need to batch audio that's going to be Mel transformed
            data = self.initial_layers(data.view(BATCH_SIZE, self.linear_units_in))
            data = data.reshape((BATCH_SIZE, 1, self.n_fft, -1))
        elif mode == 'stft':
            data = self.initial_layers(data.view(BATCH_SIZE, self.linear_units_in))
            #data = audio_to_stftgram(data.view(-1), self.initial_n_fft, self.initial_n_fft//4)
            data = data.view(BATCH_SIZE, self.n_channels, self.initial_n_fft, -1)
        elif mode == 'specgram':
            post_init = audio_to_specgram(
                self.initial_layers(
                    data.view(BATCH_SIZE, self.linear_units_in).to(self.device)
                ).view(-1)
            ).view(BATCH_SIZE, self.n_channels, -1)
        else:
            data = data.view(BATCH_SIZE, self.linear_units_in)
            data = self.initial_layers(data)
            data = data.view(BATCH_SIZE, self.n_channels, 1, -1)

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

        vv_cprint("|}} Initial layers done.")
        vv_cprint("|}} data.shape:", data.shape)

        post_conv = self.conv_layers(post_init).requires_grad_(True)
        if self.training:
            post_conv.retain_grad()

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
                

        #self.sanity_check(data)

        vv_cprint("|}} Convolution layers done.")
        vv_cprint("|}} data.shape:", data.shape)

        post_final = self.final_layers(post_conv)[-1]
        if self.training:
            post_final.retain_grad()
        #self.sanity_check(data)
        if mode == 'mel' and DIS_MODE == 2:
            ret = check_and_fix_size(data.squeeze(), self.n_fft, -2)
        elif mode == 'mel':
            invmel_inp = check_and_fix_size(data, self.n_fft, -2)
            stft = MelToSTFTWithGradients.apply(invmel_inp, self.n_fft)
            data = stft_to_audio(stft, GEN_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_PREVIEW)
        elif mode == 'stft':
            stftgram = data.view(self.n_channels, self.n_fft, -1).requires_grad_(True)
            if self.training:
                stftgram.retain_grad()
            stftgram[0, :, :] = F.normalize(stftgram[0, :, :], dim=-1)
            stftgram[1, :, :] = F.normalize(stftgram[1, :, :], dim=-1)
            data = stftgram_to_audio(stftgram, GEN_HOP_LEN)
            self.sanity_check(data)
            data = normalize_data(data)
            #self.sanity_check(data)
        elif mode == 'specgram':
            ret = normalize_data(specgram_to_audio(post_final.squeeze()[..., :TOTAL_SAMPLES_OUT]))
            if self.training:
                ret.retain_grad()
            #self.sanity_check(data)
        else:
            data = normalize_data(data.view(-1)[:TOTAL_SAMPLES_OUT])
            #self.sanity_check(data)
        vv_cprint("|}} Final layers done.")
        vv_cprint("|}} data.shape:", ret.shape)
        
        return ret

    def gen_conv(self, inputs, mode):
        assert(GEN_MODE in (1, 3, 4, 5, 6))
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        post_net = self.run_conv_net(inputs, mode)
        vv_cprint("|} Whew... Made it out of the net alive!")
        if mode == 'mel' and DIS_MODE == 2:
            return post_net
        else:
            if not torch.isfinite(post_net[0]):
                #print("Warning, got invalid output!")
                pass
            return post_net.squeeze()[:TOTAL_SAMPLES_OUT]

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
        