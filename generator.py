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
        self.ksz = GEN_KERNEL_SIZE
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

class TAGenerator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.loss = nn.BCELoss()

        if GEN_MODE == 0:
            #TODO: implement RNN/Hilbert mode
            raise NotImplementedError("TODO: implement RNN/Hilbert mode")
        elif GEN_MODE == 1:
            #TODO: port RNN/Audio code to pytorch
            raise NotImplementedError("TODO: implement RNN/Audio mode")
        elif GEN_MODE == 2:
            self.create_conv_net(mode='hilbert')
        elif GEN_MODE == 3:
            self.create_conv_net(mode='audio')
        elif GEN_MODE == 4:
            self.create_conv_net(mode='mel')
        else:
            # self.total_samp_out = TOTAL_PARAM_UPDATES * N_PARAMS
            # self.desired_process_units = TOTAL_SAMPLES_IN
            # self.create_dense_net(hilb_mode=False)
            raise ValueError("Created wrong generator class!")

    def criterion(self, label, output):
        return self.loss(output.mean().cuda(), label.mean().cuda())

    def create_conv_net(self, mode='audio'):
        self.process_layers = []

        self.sr = SAMPLE_RATE
        if mode == 'audio':
            self.n_channels = N_CHANNELS
            self.total_samp_out = TOTAL_SAMPLES_OUT
            ct = nn.ConvTranspose1d
            cv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif mode == 'mel':            
            self.n_channels = N_GEN_MEL_CHANNELS
            self.n_bins = N_GEN_FFT // 2 + 1
            self.total_samp_out = self.n_bins * TOTAL_SAMPLES_OUT
            ct = nn.ConvTranspose2d
            cv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif mode == 'hilbert':
            self.n_channels = 2
            self.total_samp_out = self.n_channels * TOTAL_SAMPLES_OUT
            ct = nn.ConvTranspose1d
            cv = nn.Conv1d
            bn = nn.BatchNorm1d
        self.n_batches = N_BATCHES

        self.padding_mode = 'reflect'

        self.stride1 = GEN_STRIDE1
        self.scale = GEN_SCALE

        v_cprint("*-"*39 + "*")
        v_cprint("Target length is", self.total_samp_out, "samples.")

        v_cprint("Creating Activation layer.")
        #self.initial_layer = nn.ReLU().cuda()
        self.initial_layers = [
            #nn.Linear(TOTAL_SAMPLES_IN, self.kernel_size*self.shuffle_fac*TOTAL_SAMPLES_IN*N_BATCHES).cuda()
            nn.ReLU(True).cuda(),
        ]
        # if mode == 'mel':
        #     v_cprint("Creating Mel Spectrogram layer.")
        #     self.initial_layers.append(torchaudio.transforms.MelSpectrogram(self.sr, TOTAL_SAMPLES_IN, n_mels=self.n_channels))
        
        self.n_processing_indices = 1
        #self.max_Lout = stride_quotient*self.total_samp_out//N_BATCHES
        #self.max_s_c = TOTAL_SAMPLES_OUT
        #Lin = TOTAL_SAMPLES_IN*GEN_SAMPLE_RATE_FACTOR//2
        s_t = self.stride1
        s_c = 2
        k_t = GEN_KERNEL_SIZE
        samps_pre_transpose = BATCH_SIZE*TOTAL_SAMPLES_IN * self.n_channels
        samps_post_transpose = samps_pre_transpose * s_t
        samps_post_conv = samps_post_transpose // s_c
        while samps_post_conv < self.total_samp_out or self.n_processing_indices < MIN_N_GEN_LAYERS+1:
            #Lin = int(Lin * self.scale)
            
            #sqrt_ksz = int(np.sqrt(self.kernel_size))

            #n_batches = int(Lout // (N_CHANNELS * self.kernel_size * BATCH_OPTIMIZATION_FACTOR))
            v_cprint("="*80)
            v_cprint("Creating ConvTranspose1d layer.", samps_pre_transpose, ">", samps_post_transpose)
            self.process_layers.append(ct(
                self.n_channels, self.n_channels, groups=1,
                kernel_size=k_t, stride=s_t, padding=0, dilation=1
            ).cuda())
            
            v_cprint("Creating Normalization layer.")
            self.process_layers.append(bn(self.n_channels).cuda())
            v_cprint("Creating Activation layer.")
            self.process_layers.append(nn.ReLU(True).cuda())
            #Lin = int(BATCH_SIZE*(2**GEN_SAMPLE_RATE_FACTOR)*(SAMPLE_RATE/44100))
            v_cprint("Creating Conv1d layer.", samps_post_transpose, ">", samps_post_conv)
            self.process_layers.append(cv(
                self.n_channels, self.n_channels, groups=1,
                kernel_size=1, stride=s_c, padding=0, dilation=1
            ).cuda())
            self.n_processing_indices += 1
            #if samps_post_conv > self.total_samp_out or self.n_processing_indices < MIN_N_GEN_LAYERS:
            v_cprint("Creating Normalization layer.")
            self.process_layers.append(bn(self.n_channels).cuda())
            v_cprint("Creating Activation layer.")
            self.process_layers.append(nn.ReLU(True).cuda())
            old_s_t = s_t
            s_t *= self.scale
            samps_pre_transpose = samps_post_conv
            samps_post_transpose = samps_pre_transpose * s_t
            samps_post_conv = samps_post_transpose // s_c
            if samps_post_conv >= self.total_samp_out:
                #quot = samps_post_conv / (self.total_samp_out)
                s_t = 1
                k_t = 1
                if mode == 'hilbert':
                    s_c = 2
                else:
                    s_c = 4
            else:
                s_t = s_t * self.scale
                if mode == 'hilbert':
                    s_c = 1
                else:
                    s_c = 2
            

            
        
        v_cprint("="*80)
        v_cprint("Created", self.n_processing_indices-1, "sets of processing layers.")

        v_cprint("Creating final layers.")
        if mode == 'mel' or mode == 'hilbert':
            self.final_layers = [
                # torchaudio.transforms.InverseMelScale(self.n_bins, self.n_channels, self.sr),
                # nn.Flatten().cuda(),
                # nn.Tanh().cuda(),
                # #torchaudio.transforms.Resample(self.sr, SAMPLE_RATE),
                # nn.Flatten().cuda(),
            ]
        else:
            self.final_layers = [
                nn.Flatten().cuda(),
                nn.Tanh().cuda(),
                #torchaudio.transforms.Resample(self.sr, SAMPLE_RATE),
                nn.Flatten().cuda(),
            ]
        
        self.initial_layers = nn.Sequential(*self.initial_layers).cuda()
        self.process_layers = nn.Sequential(*self.process_layers).cuda()
        self.final_layers = nn.Sequential(*self.final_layers).cuda()

    def create_rnn_net(self, hilb_mode=False):
        pass

    def run_conv_net(self, data, mode):
        pre_init = data.flatten().float()
        vv_cprint("|}} Initial data.shape:", pre_init.shape)
        if mode == 'mel':
            # no need to batch audio that's going to be Mel transformed
            post_init = self.initial_layers(pre_init.clone())
            pre_process = post_init.reshape(BATCH_SIZE, self.n_channels, TOTAL_SAMPLES_IN).unsqueeze(-2)
        elif mode == 'hilbert':
            #post_init = self.initial_layers(pre_init.clone()).flatten()
            #amp, phas = hilbert_from_scratch_pytorch(post_init)
            #hilb = torch.stack((amp, phas)).cuda()
            #pre_process = hilb.unsqueeze(0).unsqueeze(-1)
            post_init = self.initial_layers(pre_init.clone())
            pre_process = post_init.reshape(BATCH_SIZE, self.n_channels, TOTAL_SAMPLES_IN)
        else:
            batched = pre_init.clone().reshape((BATCH_SIZE, N_CHANNELS, TOTAL_SAMPLES_IN))
            pre_process = self.initial_layers(batched.clone())
        
        vv_cprint("|}} Initial layers done.")
        vv_cprint("|}} data.shape:", pre_process.shape)
        if mode == 'mel':
            post_process = self.process_layers(pre_process.clone())
        else:
            post_process = self.process_layers(pre_process.clone())
        vv_cprint("|}} Processing layers done.")
        vv_cprint("|}} data.shape:", post_process.shape)
        post_final = self.final_layers(post_process.clone())
        if mode == 'mel' and DIS_MODE == 2:
            ret = post_final.clone()[:, :, :, :TOTAL_SAMPLES_OUT]
        elif mode == 'hilbert':
            invhilb_inp = post_final.clone().squeeze()[:, :TOTAL_SAMPLES_OUT]
            ret = inverse_hilbert_pytorch(invhilb_inp[0], invhilb_inp[1]).flatten()#[:TOTAL_SAMPLES_OUT]
        elif mode == 'mel':
            invmel_inp = post_final.clone().squeeze()[:, :TOTAL_SAMPLES_OUT]
            ret = InverseMelWithGradients.apply(invmel_inp, N_GEN_FFT, GEN_HOP_LEN).flatten()[:TOTAL_SAMPLES_OUT]
        else:
            ret = post_final.clone().flatten()[:TOTAL_SAMPLES_OUT]
        vv_cprint("|}} Final layers done.")
        vv_cprint("|}} data.shape:", ret.shape)
        
        return ret

    def gen_conv(self, inputs, mode):
        assert(GEN_MODE in (2, 3, 4))
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        #prepped = DataBatchPrep.apply(inputs, 1, TOTAL_SAMPLES_IN, None)
        post_net = self.run_conv_net(inputs, mode)
        
        vv_cprint("|} Whew... Made it out of the net alive!")
        if mode == 'mel':
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
            return None
            #return self.gen_rnn(inputs)
        if GEN_MODE == 2:
            return self.gen_conv(inputs, 'hilbert')
        if GEN_MODE == 3:
            return self.gen_conv(inputs, 'audio')
        if GEN_MODE == 4:
            return self.gen_conv(inputs, 'mel')
        else:
            return None
    
    def forward(self, inputs):
        return self.gen_fn(inputs)