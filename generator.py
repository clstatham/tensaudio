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
        self.ndf = N_PARAMS * (5+TOTAL_PARAM_UPDATES)

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
        self.layers.append(nn.Sigmoid())

        self.layers = nn.ModuleList(self.layers)
    
    def criterion(self, label, output):
        l = self.loss(torch.unsqueeze(output, 0), torch.unsqueeze(label, 0))
        l = F.relu(l)
        return l
    
    def forward(self, inp):
        #print(inp.shape)
        out = self.layers[0](inp)
        i = 1
        for layer in self.layers[1:]:
            out = layer(out)
            i += 1
        out = out.view(out.size(0), -1)
        out -= out.min(1, keepdim=True)[0]
        out /= out.max(1, keepdim=True)[0]
        out = out.view(self.ndf)
        out = out[N_PARAMS*5:] # for some reason the first few sets of params are always 0,
                             # so we generate 5 more than we need and start at the 6th
        #print(out.shape)
        return out

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
        elif GEN_MODE == 5:
            self.total_samp_out = TOTAL_PARAM_UPDATES * N_PARAMS
            self.desired_process_units = TOTAL_SAMPLES_IN
            self.create_dense_net(hilb_mode=False)

    def criterion(self, label, output):
        l = self.loss(torch.unsqueeze(output, 0), torch.unsqueeze(label, 0))
        l = F.relu(l)
        return l
    
    def create_dense_net(self, hilb_mode=False):
        self.preprocess_layers = []
        self.process_layers = []
        self.postprocess_layers = []
        self.n_process_units = 0

        self.kernel_size = GEN_KERNEL_SIZE
        self.padding = 0
        self.dilation = 1
        self.output_padding = 0
        self.stride = 1

        v_cprint("*-"*39 + "*")

        v_cprint("Creating Linear layer.")
        #self.initial_layer = nn.ReLU().cuda()
        self.initial_layer = nn.Linear(TOTAL_SAMPLES_IN, TOTAL_SAMPLES_IN).cuda()

        self.n_preprocessing_indices = 1
        self.n_postprocessing_indices = 1

        Lout = 0
        Lin = TOTAL_SAMPLES_IN
        while Lout*N_BATCHES*N_CHANNELS*4 < self.desired_process_units:
            Lout=self.n_preprocessing_indices*(Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
            vv_cprint("Creating deconvolution layer", Lin, ">", Lout, "ksz =", self.kernel_size, "stride =", self.stride, "total =", N_BATCHES*N_CHANNELS*Lout)
            self.preprocess_layers.append(nn.ConvTranspose1d(Lin, Lout, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation).cuda())
            Lin  = Lout
            self.n_preprocessing_indices += 1
        self.n_preprocessing_indices -= 1
        v_cprint("Created", self.n_preprocessing_indices+1, "preprocessing layers.")

        if GEN_MODE in [5]:
            self.n_process_units = TOTAL_SAMPLES_IN
        else:
            self.n_process_units = Lout*N_BATCHES*N_CHANNELS*(TOTAL_SAMPLES_IN//2)
        vv_cprint("Creating", N_PROCESS_LAYERS, "Linear layers", self.n_process_units, ">", self.n_process_units)
        for i in range(N_PROCESS_LAYERS):
            self.process_layers.append(nn.Linear(self.n_process_units, self.n_process_units).cuda())
        v_cprint("Created", len(self.process_layers), "processing layers.")

        Lin = self.n_process_units
        Lout = 0
        while Lout*N_BATCHES*N_CHANNELS*self.kernel_size < self.total_samp_out:
            Lout=self.n_postprocessing_indices*(Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
            vv_cprint("Creating deconvolution layer", Lin, ">", Lout, "ksz =", self.kernel_size, "stride =", self.stride, "total =", N_BATCHES*N_CHANNELS*Lout)
            self.postprocess_layers.append(nn.ConvTranspose1d(Lin, Lout, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation).cuda())
            Lin  = Lout
            self.n_postprocessing_indices += 1
        self.n_postprocessing_indices -= 1
        v_cprint("Created", self.n_postprocessing_indices+1, "postprocessing layers.")

        v_cprint("Creating Linear Layer.")
        #self.final_layer = nn.ReLU().cuda()
        self.final_layer = nn.Linear(Lout, self.total_samp_out)
        
        self.preprocess_layers = nn.ModuleList(self.preprocess_layers).cuda()
        self.process_layers = nn.ModuleList(self.process_layers).cuda()
        self.postprocess_layers = nn.ModuleList(self.postprocess_layers).cuda()

    def gen_rnn(self, inputs):
        assert(GEN_MODE == 1)

        audio = inputs

        audio = prep_data_for_batch_operation(audio)

        if self.rnn_state is None:
            f_state1, f_state2 = self.audio_rnns[0].get_initial_state(audio)
            #f_state1 = None
            #b_state1 = self.audio_rnns[0].backward_layer.get_initial_state(audio)
            #f_state1 = f_state1[0]
        else:
            #f_state1, b_state1 = self.rnn_state
            f_state1 = self.rnn_state
        #b_state1 = b_state1[0]

        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")

        audio, f_state1, f_state2 = self.audio_rnns[0](audio, initial_state=(f_state1, f_state2), training=training)
        vv_cprint("|} Layer 1 done.")
        for i in range(N_RNN_LAYERS-1):
            audio = prep_data_for_batch_operation(audio)
            #audio = tf.expand_dims(audio, axis=0)
            audio, f_state1, f_state2 = self.audio_rnns[i+1](audio, initial_state=(f_state1, f_state2), training=training)
            vv_cprint("|}} Layer", i+2, "done.")

        vv_cprint("|} Whew... Made it out of the net alive!")

        #self.rnn_state = (f_state1, f_state2)
        audio = K.flatten(audio)
        return audio

    def run_dense_net(self, data, hilb_mode=False):
        vv_cprint("|}} Initial data.shape:", data.shape)
        data = data.flatten()
        data = self.initial_layer(data.float())

        Lin = TOTAL_SAMPLES_IN
        Lout = (Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
        data = prep_data_for_batch_operation(data, None, TOTAL_SAMPLES_IN, 1)

        vv_cprint("|}} Initial layer done.")
        vv_cprint("|}} data.shape:", data.shape)
        i = 0
        for layer in self.preprocess_layers:
            data = layer(data.float())
            vv_cprint("|}} Pre-process layer", i+1, "done.")
            vv_cprint("|}} data.shape:", data.shape)
            i += 1
        
        data = torch.squeeze(data)
        data = data.flatten()
        vv_cprint("|}} data.shape going into process layers:", data.shape)
        for i in range(len(self.process_layers)):
            data = self.process_layers[i](data)
            data = data.flatten()
            #vv_cprint("|}} Process layer", i+1, "done.")
            #vv_cprint("|}} data.shape:", data.shape)
        vv_cprint("|}} Process layers done.")
        vv_cprint("|}} data.shape:", data.shape)

        Lin = self.n_process_units
        Lout = (Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
        data = prep_data_for_batch_operation(data, None, Lin, 1)

        i = 1
        for layer in self.postprocess_layers:
            data = layer(data)
            vv_cprint("|}} Post-process layer", i, "done.")
            vv_cprint("|}} data.shape:", data.shape)
            i += 1
        
        data = data.flatten()
        #print(data.shape)
        data = self.final_layer(data)
        data = data.flatten()
        vv_cprint("|}} Final layer done.")
        vv_cprint("|}} Final data.shape:", data.shape)
        return data

    def gen_dense_hilb(self, hilb):
        assert (GEN_MODE == 2)
        hilb = torch.tensor(stack_hilb(hilb)).cuda()
        hilb = prep_data_for_batch_operation(hilb, N_BATCHES, N_CHANNELS, None)
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
        assert(GEN_MODE in (2, 3, 5))
        #inputs = tf.expand_dims(inputs, axis=0)
        #audio = my_audioert(inputs)
        #audio = audio_tensor(x[0], x[1])
        out = inputs
        orig_shape = out.shape
        
        out = prep_data_for_batch_operation(out, N_BATCHES, N_CHANNELS, None)
        vv_cprint("|} Ready for launch! Going to the net now, wheeee!")
        out = self.run_dense_net(out, hilb_mode=False)
        vv_cprint("|} Whew... Made it out of the net alive!")
        
        out = torch.squeeze(out)
        #audio = audio[:, -1, :] # the last

        out = torch.flatten(out)
        out, error = ensure_size(out, self.total_samp_out)
        if error > 0:
            vv_cprint("|} Truncated output by", error, "samples.")
        elif error < 0:
            vv_cprint("|} Padded output by", -error, "samples.")
        return out

    def gen_fn(self, inputs):
        if GEN_MODE == 0:
            amp, phase = my_hilbert(inputs)
            amp, phase = self.gen_rnn_hilb(torch.stack((amp, phase)).cuda())
            return inverse_hilbert(amp, phase)
        if GEN_MODE == 1:
            return self.gen_rnn(inputs)
        if GEN_MODE == 2:
            amp, phase = my_hilbert(inputs)
            amp, phase = self.gen_dense_hilb(torch.stack((amp, phase)).cuda())
            return inverse_hilbert(amp, phase)
        if GEN_MODE == 3:
            return self.gen_dense(inputs)
        if GEN_MODE == 5:
            o = self.gen_dense(inputs)
            #out = out.view(out.size(0), -1)
            #out -= out.min(1, keepdim=True)[0]
            #out /= out.max(1, keepdim=True)[0]
            #out = out.view(self.total_samp_out)
            return o.abs()
    
    def forward(self, inputs):
        return self.gen_fn(inputs)