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

        self.n_layers = N_GEN_LAYERS
        self.ksz = GEN_KERNEL_SIZE
        self.ndf = N_PARAMS * 2 * TOTAL_PARAM_UPDATES # ensure we never run out of params

        self.layers = [nn.ConvTranspose1d(TOTAL_SAMPLES_IN, self.ndf, self.ksz)]
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
        self.rnn_state = (
            ag.Variable(torch.randn((2*N_CHANNELS, N_BATCHES, SAMPLES_PER_BATCH)).cuda()),
            ag.Variable(torch.randn((2*N_CHANNELS, N_BATCHES, SAMPLES_PER_BATCH)).cuda()),
        )

        self.loss = nn.BCELoss()

        if GEN_MODE == 0:
            #TODO: implement RNN/Hilbert mode
            raise NotImplementedError("TODO: implement RNN/Hilbert mode")
        elif GEN_MODE == 1:
            #TODO: port RNN/Audio code to pytorch
            raise NotImplementedError("TODO: implement RNN/Audio mode")
        elif GEN_MODE == 2:
            self.create_conv_net(hilb_mode=True)
        elif GEN_MODE == 3:
            self.create_conv_net(hilb_mode=False)
        else:
            # self.total_samp_out = TOTAL_PARAM_UPDATES * N_PARAMS
            # self.desired_process_units = TOTAL_SAMPLES_IN
            # self.create_dense_net(hilb_mode=False)
            raise ValueError("Created wrong generator class!")

    def criterion(self, label, output):
        return self.loss(output.cuda(), label.cuda())

    def create_conv_net(self, hilb_mode=False):
        self.process_layers = []

        self.sr = SAMPLE_RATE * GEN_SAMPLE_RATE_FACTOR
        self.total_samp_out = TOTAL_SAMPLES_OUT * GEN_SAMPLE_RATE_FACTOR
        self.n_batches = N_BATCHES//GEN_SAMPLE_RATE_FACTOR

        self.kernel_size = GEN_KERNEL_SIZE
        self.padding = 0
        self.padding_mode = 'reflect'
        self.dilation = 1
        self.output_padding = 0
        self.stride1 = 4
        self.stride2 = 1
        self.scale1 = 4
        self.scale2 = 1.1
        self.n_rnn = 2

        v_cprint("*-"*39 + "*")
        v_cprint("Target length is", self.total_samp_out, "samples.")

        v_cprint("Creating Activation layer.")
        #self.initial_layer = nn.ReLU().cuda()
        self.initial_layers = [
            #nn.Linear(TOTAL_SAMPLES_IN, self.kernel_size*self.shuffle_fac*TOTAL_SAMPLES_IN*N_BATCHES).cuda()
            nn.ReLU(inplace=True).cuda(),
        ]

        self.n_processing_indices = 1
        self.max_Lout = self.stride2*self.total_samp_out//(self.n_batches*self.stride1)
        Lin = TOTAL_SAMPLES_IN*GEN_SAMPLE_RATE_FACTOR
        Lout = Lin
        while Lout < self.max_Lout or self.n_processing_indices < N_GEN_LAYERS:
            #Lin = int(Lin * self.scale)
            Lout = int(self.scale1 * Lin)
            Lout = int(min(Lout, self.max_Lout))
            Lout_scaled2 = int(Lout * self.scale2)
            Lout_scaled2_strided1 = Lout_scaled2 * self.stride1
            Lout_scaled2_strided2 = Lout_scaled2 * self.stride2
            #sqrt_ksz = int(np.sqrt(self.kernel_size))

            #n_batches = int(Lout // (N_CHANNELS * self.kernel_size * BATCH_OPTIMIZATION_FACTOR))
            v_cprint("="*80)
            v_cprint("Creating ConvTranspose1d layer.", Lin, ">", Lout_scaled2)
            self.process_layers.append(nn.ConvTranspose1d(Lin, Lout_scaled2, groups=1, kernel_size=self.kernel_size, stride=self.stride1, padding=self.padding, dilation=self.dilation).cuda())
            
            v_cprint("Creating Normalization layer.")
            self.process_layers.append(nn.BatchNorm1d(Lout_scaled2).cuda())
            v_cprint("Creating Activation layer.")
            self.process_layers.append(nn.ReLU().cuda())
            #Lin = int(BATCH_SIZE*(2**GEN_SAMPLE_RATE_FACTOR)*(SAMPLE_RATE/44100))
            Lin  = Lout*self.stride2
            v_cprint("Creating Conv1d layer.", Lout_scaled2, ">", Lin)
            self.process_layers.append(nn.Conv1d(Lout_scaled2, Lin, groups=1, kernel_size=1, stride=self.stride2, padding=0, dilation=1).cuda())
            v_cprint("Creating Normalization layer.")
            self.process_layers.append(nn.BatchNorm1d(Lin).cuda())
            v_cprint("Creating Activation layer.")
            self.process_layers.append(nn.ReLU().cuda())
            
            

            self.n_processing_indices += 1
        
        v_cprint("="*80)
        v_cprint("Created", self.n_processing_indices-1, "sets of processing layers.")

        v_cprint("Creating final layers.")
        self.final_layers = [
            nn.Flatten().cuda(),
            torchaudio.transforms.Resample(self.sr, SAMPLE_RATE),            
            nn.Tanh().cuda(),
            nn.Flatten().cuda(),
        ]
        
        self.initial_layers = nn.Sequential(*self.initial_layers).cuda()
        self.process_layers = nn.Sequential(*self.process_layers).cuda()
        self.final_layers = nn.Sequential(*self.final_layers).cuda()

    def create_rnn_net(self, hilb_mode=False):
        pass

    def run_conv_net(self, data, hilb_mode=False):
        pre_init = data.flatten()
        vv_cprint("|}} Initial data.shape:", pre_init.shape)
        post_init = self.initial_layers.forward(pre_init.float())

        batched = ag.Variable(post_init.clone().reshape((self.n_batches, GEN_SAMPLE_RATE_FACTOR*TOTAL_SAMPLES_IN, self.kernel_size)))
        vv_cprint("|}} Initial layers done.")
        vv_cprint("|}} data.shape:", batched.shape)
        post_process = ag.Variable(self.process_layers.forward(batched).flatten()[:self.total_samp_out])
        vv_cprint("|}} Processing layers done.")
        vv_cprint("|}} data.shape:", post_process.shape)

        #batched_rnn = ag.Variable(post_process.clone().reshape((2*N_CHANNELS, N_BATCHES, SAMPLES_PER_BATCH)))
        
        #post_rnn, self.rnn_state = self.rnn_layers(batched_rnn, self.rnn_state)
        #vv_cprint("|}} RNN layers done.")
        #vv_cprint("|}} data.shape:", post_rnn.shape)

        # post_final = self.final_layers.forward(post_rnn).flatten()
        # vv_cprint("|}} Final layer done.")
        # vv_cprint("|}} Final data.shape:", post_final.shape)
        return post_process.flatten()

    def gen_dense(self, inputs):
        assert(GEN_MODE in (2, 3))
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        #prepped = DataBatchPrep.apply(inputs, 1, TOTAL_SAMPLES_IN, None)
        post_net = self.run_conv_net(inputs, hilb_mode=False)[:TOTAL_SAMPLES_OUT].squeeze()
        if not torch.isfinite(post_net[0]):
            print("Warning, got invalid output!")
        vv_cprint("|} Whew... Made it out of the net alive!")
        
        return normalize_data(post_net[:TOTAL_SAMPLES_OUT])

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
            return None
        if GEN_MODE == 3:
            return self.gen_dense(inputs)
        else:
            return None
    
    def forward(self, inputs):
        return self.gen_fn(inputs)