import six
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.rnn_state = None
        self.total_samp_out = TOTAL_SAMPLES_OUT
        self.desired_process_units = DESIRED_PROCESS_UNITS
        self.loss = nn.BCELoss()

        if GEN_MODE == 0:
            #TODO: implement RNN/Hilbert mode
            raise NotImplementedError("TODO: implement RNN/Hilbert mode")
        elif GEN_MODE == 1:
            #TODO: port RNN/Audio code to pytorch
            raise NotImplementedError("TODO: implement RNN/Audio mode")
        elif GEN_MODE == 2:
            self.create_dense_net(hilb_mode=True)
        elif GEN_MODE == 3:
            self.create_dense_net(hilb_mode=False)
        else:
            # self.total_samp_out = TOTAL_PARAM_UPDATES * N_PARAMS
            # self.desired_process_units = TOTAL_SAMPLES_IN
            # self.create_dense_net(hilb_mode=False)
            raise ValueError("Created wrong generator class!")

    def criterion(self, label, output):
        l = self.loss(torch.unsqueeze(output.cuda(), 0), torch.unsqueeze(label, 0))
        l = F.relu(l)
        return l
    
    def create_dense_net(self, hilb_mode=False):
        self.preprocess_layers = []
        self.process_layers = []
        self.postprocess_layers = []
        self.n_process_units = 0

        self.kernel_size = GEN_KERNEL_SIZE
        self.padding = 0
        self.padding_mode = 'zeros'
        self.dilation = 8
        self.output_padding = 0
        self.stride = 1

        v_cprint("*-"*39 + "*")

        v_cprint("Creating Linear layer.")
        #self.initial_layer = nn.ReLU().cuda()
        self.initial_layers = [
            nn.Linear(TOTAL_SAMPLES_IN, TOTAL_SAMPLES_IN).cuda()
        ]

        self.n_preprocessing_indices = 1
        self.n_postprocessing_indices = 1

        Lout = 0
        Lin = TOTAL_SAMPLES_IN
        magic = 1 # TODO: get rid of this
        n_batches = 1
        while Lout < self.desired_process_units:
            Lout=2*((Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1)**(2*self.n_preprocessing_indices)
            vv_cprint("Creating deconvolution layer", Lin, ">", Lout, "ksz =", self.kernel_size, "stride =", self.stride, "total =", n_batches*DESIRED_PROCESS_UNITS_FACTOR*Lout)            
            Lout = int(min(Lout, self.desired_process_units))
            self.preprocess_layers.append(nn.ConvTranspose1d(Lin, Lout, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, padding_mode=self.padding_mode, output_padding=self.output_padding, dilation=self.dilation).cuda())
            Lin  = Lout
            self.n_preprocessing_indices += 1
        #self.preprocess_layers.append(nn.Flatten())
        v_cprint("Created", self.n_preprocessing_indices+1, "preprocessing layers.")

        n_batches = int(Lout // (N_CHANNELS * self.kernel_size * BATCH_OPTIMIZATION_FACTOR))
        #self.n_process_units = Lout * n_batches * DESIRED_PROCESS_UNITS_FACTOR
        #assert(self.n_process_units <= self.desired_process_units)
        #print("Creating", N_PROCESS_LAYERS, "Linear layers", self.n_process_units, ">", self.n_process_units)
        #for i in range(N_PROCESS_LAYERS):
            #self.process_layers.append(nn.Linear(self.n_process_units, self.n_process_units).cuda())
        #self.process_layers.append(nn.BatchNorm1d(Lout))
        v_cprint("Created", len(self.process_layers), "processing layers.")

        Lin = Lout
        Lout = 0
        while Lout*N_BATCHES < TOTAL_SAMPLES_OUT:
            Lout=2*((Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1)**(2*self.n_postprocessing_indices)
            #n_batches = int(Lout // (N_CHANNELS * self.kernel_size * BATCH_OPTIMIZATION_FACTOR))
            Lout = int(min(Lout, TOTAL_SAMPLES_OUT))
            vv_cprint("Creating deconvolution layer", Lin, ">", Lout, "ksz =", self.kernel_size, "stride =", self.stride, "total =", N_BATCHES*N_CHANNELS*Lout)
            self.postprocess_layers.append(nn.ConvTranspose1d(Lin, Lout, kernel_size=1, stride=self.stride, padding=self.padding, padding_mode=self.padding_mode, output_padding=self.output_padding, dilation=self.dilation).cuda())
            Lin  = Lout
            self.n_postprocessing_indices += 1
        #self.postprocess_layers.append(nn.ConvTranspose1d(Lout, int(Lout*2), kernel_size=1, stride=1))
        Lout *= int(N_BATCHES)
        assert(Lout >= self.total_samp_out)
        v_cprint("Created", self.n_postprocessing_indices+1, "postprocessing layers.")

        v_cprint("Creating final layer.")
        #self.final_layer = nn.ReLU().cuda()
        #Lout=self.n_postprocessing_indices*(Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
        #assert(Lout*N_BATCHES >= int(self.total_samp_out) and Lout*N_BATCHES <= int(self.total_samp_out*2))
        self.final_layers = [
            nn.Flatten(),
        ]
        
        self.initial_layers = nn.Sequential(*self.initial_layers).cuda()
        self.preprocess_layers = nn.Sequential(*self.preprocess_layers).cuda()
        self.process_layers = nn.Sequential(*self.process_layers).cuda()
        self.postprocess_layers = nn.Sequential(*self.postprocess_layers).cuda()
        self.final_layers = nn.Sequential(*self.final_layers).cuda()

    def run_dense_net(self, data, hilb_mode=False):
        pre_init = data.flatten()
        vv_cprint("|}} Initial data.shape:", pre_init.shape)
        post_init = self.initial_layers.forward(pre_init.float())

        pre_preprocess, _ = DataBatchPrep.apply(data.clone(), None, TOTAL_SAMPLES_IN, 1)

        vv_cprint("|}} Initial layers done.")
        vv_cprint("|}} data.shape:", pre_preprocess.shape)
        post_preprocess = self.preprocess_layers.forward(pre_preprocess)
        #pre_process, _ = DataBatchPrep.apply(post_preprocess.clone(), BAT, 1, None)
        #vv_cprint("|}} data.shape going into process layers:", pre_process.shape)
        #post_process = self.process_layers.forward(pre_process)
        #vv_cprint("|}} Process layers done.")
        #vv_cprint("|}} data.shape:", post_process.shape)

        pre_postprocess, _ = DataBatchPrep.apply(post_preprocess.clone(), None, self.desired_process_units, 1)

        post_postprocess = self.postprocess_layers.forward(pre_postprocess)
        vv_cprint("|}} Postprocess layers done.")
        vv_cprint("|}} data.shape:", post_postprocess.shape)
        
        post_final = self.final_layers.forward(post_postprocess).flatten()
        vv_cprint("|}} Final layer done.")
        vv_cprint("|}} Final data.shape:", post_final.shape)
        return post_final

    def gen_dense_hilb(self, hilb):
        assert (GEN_MODE == 2)
        hilb = torch.tensor(stack_hilb(hilb)).cuda()
        hilb = DataBatchPrep.apply(hilb, N_BATCHES, N_CHANNELS, None)
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        hilb = self.run_dense_net(hilb, hilb_mode=True)
        vv_cprint("|} Whew... Made it out of the net alive!")
        #hilb = tf.squeeze(hilb)
        #hilb = hilb[-1, :, :]
        vv_cprint("|}} hilb.shape:", hilb.shape)
        hilb = torch.flatten(hilb)
        hilb, error = ensure_size(hilb, N_CHANNELS*TOTAL_SAMPLES_IN)
        hilb = [[hilb[i] for i in range(len(hilb)) if i%2==0], [hilb[i+1] for i in range(len(hilb)) if i%2==0]]
        if error > 0:
            vv_cprint("|} Truncated output audio by", error//N_CHANNELS, "samples.")
        else:
            vv_cprint("|} Padded output audio by", -error//N_CHANNELS, "samples.")
        #assert(len(hilb) == 2 and len(hilb[0] == TOTAL_SAMPLES_IN))
        return hilb[0], hilb[1]

    def gen_dense(self, inputs):
        assert(GEN_MODE in (2, 3))
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        prepped, _ = DataBatchPrep.apply(inputs, 1, TOTAL_SAMPLES_IN, None)
        post_net = self.run_dense_net(prepped, hilb_mode=False)[:self.total_samp_out].squeeze()
        if not torch.is_nonzero(post_net[0]):
            print("Warning, got invalid output!")
        vv_cprint("|} Whew... Made it out of the net alive!")
        return normalize_audio(post_net[:self.total_samp_out])

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
            amp, phase = my_hilbert(inputs)
            amp, phase = self.gen_dense_hilb(torch.stack((amp, phase)).cuda())
            return inverse_hilbert(amp, phase)
        if GEN_MODE == 3:
            return self.gen_dense(inputs)
        else:
            return None
    
    def forward(self, inputs):
        return self.gen_fn(inputs)