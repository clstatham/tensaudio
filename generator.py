import six
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchaudio

from helper import *
from hilbert import *
from global_constants import *

# foo = np.arange(0, N_BATCHES*2)
# cprint(foo)
# foo = prep_audio_for_batch_operation(foo)
# cprint(foo)
# foo = flatten_audio(foo)
# cprint(foo)

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
    
    def criterion(self, label, output):
        l = self.loss(output.cuda(), label)
        #l = F.relu(l)
        return l
    
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
        self.layer = nn.LSTM(in_features, hidden_features, 1, batch_first=True).cuda()
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

class TAGenerator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.loss = nn.BCELoss()

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
        elif GEN_MODE == 6:
            self.create_conv_net(mode='specgram')
        else:
            # self.total_samp_out = TOTAL_PARAM_UPDATES * N_PARAMS
            # self.desired_process_units = TOTAL_SAMPLES_IN
            # self.create_dense_net(hilb_mode=False)
            raise ValueError("Created wrong generator class!")

    def criterion(self, label, output):
        return self.loss(output.cuda(), label.cuda())

    def create_conv_net(self, mode='audio'):
        self.conv_layers = []
        self.lin_layers = []

        self.sr = SAMPLE_RATE
        if mode == 'rnnaudio':
            self.n_channels = N_CHANNELS
            self.total_samp_out = TOTAL_SAMPLES_OUT
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_SCALE_LIN * self.n_channels
            us = TAUpscalingLayer
            ds = TADownscalingLayer
            bn = nn.BatchNorm1d
        if mode == 'audio':
            self.n_channels = N_CHANNELS
            self.total_samp_out = TOTAL_SAMPLES_OUT
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels
            self.linear_units_out = TOTAL_SAMPLES_IN * GEN_SCALE_LIN * self.n_channels
            us = nn.ConvTranspose1d
            ds = nn.Conv1d
            bn = nn.BatchNorm1d
        elif mode == 'mel':
            self.n_channels = N_GEN_MEL_CHANNELS
            self.n_bins = N_GEN_FFT // 2 + 1
            self.linear_units_in = TOTAL_SAMPLES_IN * self.n_channels
            self.linear_units_out = (TOTAL_SAMPLES_IN * GEN_SCALE_LIN) * self.n_channels
            #self.total_samp_out = BATCH_SIZE * self.n_channels * TOTAL_SAMPLES_OUT // GEN_HOP_LEN
            self.total_samp_out = self.n_bins
            us = nn.ConvTranspose1d
            ds = nn.Conv1d
            bn = nn.BatchNorm1d
        elif mode == 'hilbert':
            self.n_channels = 2
            self.total_samp_out = self.n_channels * TOTAL_SAMPLES_OUT
            us = nn.ConvTranspose1d
            ds = nn.Conv1d
            bn = nn.BatchNorm1d
        elif mode == 'specgram':
            self.n_channels = 2
            self.n_bins = N_GEN_FFT//2 + 1
            self.linear_units_in = TOTAL_SAMPLES_IN**2 * self.n_channels
            self.linear_units_out = (TOTAL_SAMPLES_IN * GEN_SCALE_LIN)**2 * self.n_channels
            self.total_samp_out = self.n_bins**2
            us = nn.ConvTranspose2d
            ds = nn.Conv2d
            bn = nn.BatchNorm2d
        lin = nn.Linear

        self.padding_mode = 'reflect'

        v_cprint("*-"*39 + "*")
        v_cprint("Target length is", self.total_samp_out, "samples.")

        
        #self.initial_layer = nn.ReLU().cuda()
        self.initial_layers = [
            nn.ReLU(True).cuda(),
            nn.Linear(self.linear_units_in, self.linear_units_out).cuda(),
        ]
        v_cprint("Created Activation layer.")
        v_cprint("Created Linear layer. ", self.linear_units_in, ">", self.linear_units_out)
        
        # if mode == 'mel':
        #     v_cprint("Created Mel Spectrogram layer.")
        #     self.initial_layers.append(torchaudio.transforms.MelSpectrogram(self.sr, TOTAL_SAMPLES_IN, n_mels=self.n_channels))
        
        self.n_sets = 0
        s_us = GEN_STRIDE_UPSCALING
        k_us = GEN_KERNEL_SIZE_UPSCALING
        s_ds = GEN_STRIDE_DOWNSCALING
        k_ds = GEN_KERNEL_SIZE_DOWNSCALING
        samps_post_ds = self.linear_units_out
        while samps_post_ds < self.total_samp_out or self.n_sets < MIN_N_GEN_LAYERS+1:
            self.n_sets += 1
            #n = int(self.n_channels * s_us**(2*self.i))
            #c = int(self.n_channels * s_us**(2*self.i+1))
            k_us = GEN_KERNEL_SIZE_UPSCALING * self.n_sets + 5
            k_ds = GEN_KERNEL_SIZE_DOWNSCALING * self.n_sets + 3
            n = self.n_channels
            c = self.n_channels
            v_cprint("="*80)

            self.conv_layers.append(nn.ReLU(True).cuda())
            v_cprint("Created Activation layer.")
            
            #if mode == 'rnnaudio':
                #v_cprint("Created RNN layer.", samps_post_ds, "=", samps_post_ds)
                #self.conv_layers.append(us(samps_post_ds, samps_post_ds).cuda())
            
            
            if mode == 'specgram':
                samps_post_us = (s_us**2) * samps_post_ds
                self.conv_layers.append(us(c, n, (k_us, k_us), (s_us, s_us), bias=False).cuda())
            else:
                samps_post_us = s_us * samps_post_ds
                self.conv_layers.append(us(c, n, k_us, s_us, bias=False).cuda())
            v_cprint("Created Upscaling layer.", samps_post_ds, "<", samps_post_us)
            
            
            self.conv_layers.append(nn.ReLU(True).cuda())
            v_cprint("Created Activation layer.")
            self.conv_layers.append(bn(n).cuda())
            v_cprint("Created Normalization layer with", n, "channels.")
            
            if mode == 'specgram':
                samps_post_ds = samps_post_us // (s_ds**2)
                self.conv_layers.append(ds(n, c, (k_ds, k_ds), (s_ds, s_ds), bias=False).cuda())
            else:
                samps_post_ds = samps_post_us // s_ds
                self.conv_layers.append(ds(n, c, k_ds, s_ds, bias=False).cuda())
            v_cprint("Created Downscaling layer.", samps_post_us, ">", samps_post_ds)
            
            self.conv_layers.append(bn(c).cuda())
            v_cprint("Created Normalization layer with", c, "channels.")

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

        v_cprint("Created final layers.")
        if mode == 'mel' or mode == 'hilbert' or mode == 'specgram':
            self.final_layers = [
                # torchaudio.transforms.InverseMelScale(self.n_bins, self.n_channels, self.sr),
                # nn.Flatten().cuda(),
                #nn.ReLU(True).cuda(),
                nn.Identity(),
                # #torchaudio.transforms.Resample(self.sr, SAMPLE_RATE),
                # nn.Flatten().cuda(),
            ]
        else:
            self.final_layers = [
                nn.Tanh().cuda(),
                #torchaudio.transforms.Resample(self.sr, SAMPLE_RATE),
            ]
        
        self.initial_layers = nn.Sequential(*self.initial_layers).cuda()
        self.lin_layers = nn.ModuleList(self.lin_layers).cuda()
        self.conv_layers = nn.ModuleList(self.conv_layers).cuda()
        self.final_layers = nn.Sequential(*self.final_layers).cuda()

    def run_conv_net(self, data, mode):
        pre_init = data.float()
        vv_cprint("|}} Initial data.shape:", pre_init.shape)
        if mode == 'mel':
            # no need to batch audio that's going to be Mel transformed
            post_init = pre_init.view(BATCH_SIZE, self.linear_units_in)
            post_usds = self.initial_layers(post_init.clone()).view(BATCH_SIZE, self.n_channels, -1)
        elif mode == 'hilbert':
            #post_init = self.initial_layers(pre_init.clone()).flatten()
            #amp, phas = hilbert_from_scratch_pytorch(post_init)
            #hilb = torch.stack((amp, phas)).cuda()
            #pre_conv = hilb.unsqueeze(0).unsqueeze(-1)
            post_init = self.initial_layers(pre_init.clone())
            post_usds = post_init.reshape((BATCH_SIZE, self.n_channels, TOTAL_SAMPLES_IN))
        elif mode == 'specgram':
            post_init = self.initial_layers(pre_init.clone().view(BATCH_SIZE, self.n_channels * TOTAL_SAMPLES_IN * TOTAL_SAMPLES_IN))
            post_usds = post_init.reshape((BATCH_SIZE, self.n_channels, TOTAL_SAMPLES_IN*GEN_SCALE_LIN, TOTAL_SAMPLES_IN*GEN_SCALE_LIN))
        else:
            post_init = pre_init.view(BATCH_SIZE, N_CHANNELS * TOTAL_SAMPLES_IN)
            post_usds = self.initial_layers(post_init.clone())
        
        vv_cprint("|}} Initial layers done.")
        vv_cprint("|}} data.shape:", post_usds.shape)
        for j in range(len(self.conv_layers)):
            #dim = int(np.sqrt(pre_conv.shape[-2]))
            pre_usds = post_usds.view(BATCH_SIZE, self.n_channels, -1)
            if mode == 'specgram':
                dim = int(np.sqrt(pre_usds.shape[-1]))
                post_us = pre_usds.view(BATCH_SIZE, self.n_channels, dim, -1)
                #post_us = F.interpolate(pre_us, scale_factor=GEN_SCALE_UPSCALING, mode='bilinear')
            else:
                post_us = pre_usds
                #post_us = F.interpolate(pre_us, scale_factor=GEN_SCALE_UPSCALING, mode='linear')
            
            post_usds = self.conv_layers[j](post_us)
            
            
        #post_process = self.conv_layers(pre_conv.clone())
        vv_cprint("|}} Convolution layers done.")
        vv_cprint("|}} data.shape:", post_usds.shape)

        #pre_final_lin = post_usds.view(BATCH_SIZE, -1)[-1, :self.total_samp_out].unsqueeze(0)
        #post_final_lin = self.lin_layers[0](pre_final_lin).view(1, 2, self.n_bins, -1)
        if BATCH_SIZE > 1:
            post_final_lin = post_usds[-1]
        else:
            post_final_lin = post_usds.squeeze()

        post_final = self.final_layers(post_final_lin.clone())
        if mode == 'mel' and DIS_MODE == 2:
            ret = post_final.clone()[:, :self.n_bins]
        elif mode == 'hilbert':
            invhilb_inp = post_final.clone().squeeze()[:, :TOTAL_SAMPLES_OUT]
            ret = inverse_hilbert_pytorch(invhilb_inp[0], invhilb_inp[1]).flatten()#[:TOTAL_SAMPLES_OUT]
        elif mode == 'mel':
            invmel_inp = post_final.clone().squeeze()[:, :self.n_bins]
            ret = MelToSTFTWithGradients.apply(invmel_inp, N_GEN_FFT, GEN_HOP_LEN).flatten()[:TOTAL_SAMPLES_OUT]
        elif mode == 'specgram':
            pre_pre_stft_inp = post_final.clone().squeeze().view(2, -1)
            total_samples = pre_pre_stft_inp.shape[-1]
            diff = self.n_bins**2 - total_samples
            if diff > 0:
                v_cprint("Warning, zero-padding output by", diff, "samples.")
                v_cprint("Number of samples in output:", total_samples)
                padding = torch.zeros((pre_pre_stft_inp.shape[0], pre_pre_stft_inp.shape[1], diff)).cuda()
                pre_stft_inp = torch.cat((pre_pre_stft_inp, padding), dim=-1).view(2, self.n_bins, -1)
            elif diff < 0:
                pre_stft_inp = pre_pre_stft_inp[..., :diff].view(2, self.n_bins, -1)
            else:
                pre_stft_inp = pre_pre_stft_inp
            
            stft_inp = DataBatchPrep.apply(pre_stft_inp, 2, self.n_bins, None)[:, :, :self.n_bins]
            stft = specgram_to_stft(stft_inp)
            ret = stft_to_audio(stft, GEN_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_PREVIEW)
        else:
            ret = post_final.clone().flatten()[:TOTAL_SAMPLES_OUT]
        vv_cprint("|}} Final layers done.")
        vv_cprint("|}} data.shape:", ret.shape)
        
        return ret

    def gen_conv(self, inputs, mode):
        assert(GEN_MODE in (2, 3, 4, 6))
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        #prepped = DataBatchPrep.apply(inputs, 1, TOTAL_SAMPLES_IN, None)
        post_net = self.run_conv_net(inputs, mode)
        
        vv_cprint("|} Whew... Made it out of the net alive!")
        if mode in ['mel', 'specgram']:
            return post_net
        else:
            if not torch.isfinite(post_net[0]):
                print("Warning, got invalid output!")
            return normalize_audio(post_net[:TOTAL_SAMPLES_OUT].squeeze())

    def gen_fn(self, inputs):
        if GEN_MODE == 0:
            #amp, phase = my_hilbert(inputs)
            #amp, phase = self.gen_rnn_hilb(torch.stack((amp, phase)).cuda())
            #return inverse_hilbert(amp, phase)
            return None
        if GEN_MODE == 1:
            return self.gen_conv(inputs, 'rnnaudio')
            #return self.gen_rnn(inputs)
        if GEN_MODE == 2:
            return self.gen_conv(inputs, 'hilbert')
        if GEN_MODE == 3:
            return self.gen_conv(inputs, 'audio')
        if GEN_MODE == 4:
            return self.gen_conv(inputs, 'mel')
        if GEN_MODE == 6:
            return self.gen_conv(inputs, 'specgram')
        else:
            return None
    
    def forward(self, inputs):
        return self.gen_fn(inputs)