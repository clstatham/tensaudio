import six
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import *
from hilbert import *
from global_constants import *

# foo = np.arange(0, N_BATCHES*2)
# print(foo)
# foo = prep_audio_for_batch_operation(foo)
# print(foo)
# foo = flatten_audio(foo)
# print(foo)

class TA_Generator(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rnn_state = None

        self.loss = nn.BCELoss()

        if GEN_MODE == 0:
            #TODO: implement RNN/Hilbert mode
            raise NotImplementedError("TODO: implement RNN/Hilbert mode")
        if GEN_MODE == 1:
            #TODO: port RNN/Audio code to pytorch
            raise NotImplementedError("TODO: implement RNN/Hilbert mode")
            self.audio_rnns = []
            self.audio_rnns.append(nn.LSTM(N_TIMESTEPS_PER_KERNEL, return_state=True, stateful=True, time_major=False, go_backwards=False, name="f_audio_rnn_0"))
            for i in range(N_RNN_LAYERS-1):
                layer_f = LSTM(N_TIMESTEPS_PER_KERNEL, return_state=True, stateful=True, time_major=False, go_backwards=False, name="f_audio_rnn_"+str(i+1))
                #layer_b = SimpleRNN(N_UNITS*2, return_state=True, stateful=True, time_major=True, go_backwards=True, name="b_audio_rnn_"+str(l))
                #self.audio_rnns.append(Bidirectional(layer_f, backward_layer=layer_b, merge_mode='ave'))
                self.audio_rnns.append(layer_f)
        elif GEN_MODE == 2:
            self.create_dense_net(hilb_mode=True)
        elif GEN_MODE == 3:
            self.create_dense_net(hilb_mode=False)
        

        # if self.manager.latest_checkpoint:
        #     self.ckpt.restore(self.manager.latest_checkpoint)
        #     print("Generator Restored from {}".format(self.manager.latest_checkpoint))
        # else:
        #     print("Generator Initializing from scratch.")

    def criterion(self, ex, op):
        return self.loss(torch.unsqueeze(torch.mean(op), 0), torch.unsqueeze(torch.mean(ex), 0))
    
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

        v_print("*-"*39 + "*")

        v_print("Creating ReLU layer.")
        self.initial_layer = nn.ReLU().cuda()

        self.n_preprocessing_indices = 1
        self.n_postprocessing_indices = 1

        Lout = 0
        Lin = TOTAL_SAMPLES_IN
        while Lout*N_BATCHES*N_CHANNELS*4 < DESIRED_PROCESS_UNITS:
            Lout=self.n_preprocessing_indices*(Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
            vv_print("Creating deconvolution layer", Lin, ">", Lout, "ksz =", self.kernel_size, "stride =", self.stride, "total =", N_BATCHES*N_CHANNELS*Lout)
            self.preprocess_layers.append(nn.ConvTranspose1d(Lin, Lout, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation).cuda())
            Lin  = Lout
            self.n_preprocessing_indices += 1
        self.n_preprocessing_indices -= 1
        v_print("Created", self.n_preprocessing_indices+1, "preprocessing layers.")

        self.n_process_units = Lout*N_BATCHES*N_CHANNELS*4
        vv_print("Creating", N_PROCESS_LAYERS, "Linear layers", self.n_process_units, ">", self.n_process_units)
        for i in range(N_PROCESS_LAYERS):
            self.process_layers.append(nn.Linear(self.n_process_units, self.n_process_units).cuda())
        print("Created", len(self.process_layers), "processing layers.")

        Lin = self.n_process_units
        Lout = 0
        while Lout*N_BATCHES*N_CHANNELS < TOTAL_SAMPLES_OUT:
            Lout=self.n_postprocessing_indices*(Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
            vv_print("Creating deconvolution layer", Lin, ">", Lout, "ksz =", self.kernel_size, "stride =", self.stride, "total =", N_BATCHES*N_CHANNELS*Lout)
            self.postprocess_layers.append(nn.ConvTranspose1d(Lin, Lout, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation).cuda())
            Lin  = Lout
            self.n_postprocessing_indices += 1
        self.n_postprocessing_indices -= 1
        v_print("Created", self.n_postprocessing_indices+1, "postprocessing layers.")

        print("Creating ReLU Layer.")
        self.final_layer = nn.ReLU().cuda()
        
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

        vv_print("|} Ready for launch! Going to the net now, wheeee!")

        audio, f_state1, f_state2 = self.audio_rnns[0](audio, initial_state=(f_state1, f_state2), training=training)
        vv_print("|} Layer 1 done.")
        for i in range(N_RNN_LAYERS-1):
            audio = prep_data_for_batch_operation(audio)
            #audio = tf.expand_dims(audio, axis=0)
            audio, f_state1, f_state2 = self.audio_rnns[i+1](audio, initial_state=(f_state1, f_state2), training=training)
            vv_print("|}} Layer", i+2, "done.")

        vv_print("|} Whew... Made it out of the net alive!")

        #self.rnn_state = (f_state1, f_state2)
        audio = K.flatten(audio)
        return audio

    def run_dense_net(self, data, hilb_mode=False):
        vv_print("|}} Initial data.shape:", data.shape)
        
        data = self.initial_layer(data.float())

        Lin = TOTAL_SAMPLES_IN
        Lout = (Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
        data = prep_data_for_batch_operation(data, None, TOTAL_SAMPLES_IN, 1)

        vv_print("|}} Initial layer done.")
        vv_print("|}} data.shape:", data.shape)
        i = 0
        for layer in self.preprocess_layers:
            data = layer(data.float())
            vv_print("|}} Pre-process layer", i+1, "done.")
            vv_print("|}} data.shape:", data.shape)
            i += 1
        
        data = torch.squeeze(data)
        data = data.flatten()
        vv_print("|}} data.shape going into process layers:", data.shape)
        for i in range(len(self.process_layers)):
            data = self.process_layers[i](data)
            data = data.flatten()
            #vv_print("|}} Process layer", i+1, "done.")
            #vv_print("|}} data.shape:", data.shape)
        vv_print("|}} Process layers done.")
        vv_print("|}} data.shape:", data.shape)

        Lin = self.n_process_units
        Lout = (Lin-1)*self.stride-2*self.padding*self.dilation*(self.kernel_size-1)*self.output_padding+1
        data = prep_data_for_batch_operation(data, None, Lin, 1)

        i = 1
        for layer in self.postprocess_layers:
            data = layer(data)
            vv_print("|}} Post-process layer", i, "done.")
            vv_print("|}} data.shape:", data.shape)
            i += 1
        
        data = self.final_layer(data)
        data = data.flatten()
        vv_print("|}} Final layer done.")
        vv_print("|}} Final data.shape:", data.shape)
        return data

    def gen_dense_hilb(self, hilb):
        assert (GEN_MODE == 2)
        hilb = torch.tensor(stack_hilb(hilb)).cuda()
        hilb = prep_data_for_batch_operation(hilb, N_BATCHES, N_CHANNELS, None)
        vv_print("|} Ready for launch! Going to the net now, wheeee!")
        hilb = self.run_dense_net(hilb, hilb_mode=True)
        vv_print("|} Whew... Made it out of the net alive!")
        #hilb = tf.squeeze(hilb)
        #hilb = hilb[-1, :, :]
        vv_print("|}} hilb.shape:", hilb.shape)
        hilb = torch.flatten(hilb)
        hilb, error = ensure_size(hilb, N_CHANNELS*TOTAL_SAMPLES_IN)
        hilb = [[hilb[i] for i in range(len(hilb)) if i%2==0], [hilb[i+1] for i in range(len(hilb)) if i%2==0]]
        if error > 0:
            vv_print("|} Truncated output audio by", error//N_CHANNELS, "samples.")
        else:
            vv_print("|} Padded output audio by", -error//N_CHANNELS, "samples.")
        #assert(len(hilb) == 2 and len(hilb[0] == TOTAL_SAMPLES_IN))
        return hilb[0], hilb[1]

    def gen_dense(self, inputs):
        assert(GEN_MODE in (2, 3))
        #inputs = tf.expand_dims(inputs, axis=0)
        #audio = my_audioert(inputs)
        #audio = audio_tensor(x[0], x[1])
        audio = inputs
        orig_shape = audio.shape
        
        audio = prep_data_for_batch_operation(audio, N_BATCHES, N_CHANNELS, None)
        vv_print("|} Ready for launch! Going to the net now, wheeee!")
        audio = self.run_dense_net(audio, hilb_mode=False)
        vv_print("|} Whew... Made it out of the net alive!")
        
        audio = torch.squeeze(audio)
        #audio = audio[:, -1, :] # the last

        audio = torch.flatten(audio)
        audio, error = ensure_size(audio, TOTAL_SAMPLES_OUT)
        if error > 0:
            vv_print("|} Truncated output audio by", error, "samples.")
        else:
            vv_print("|} Padded output audio by", -error, "samples.")
        return audio

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
    
    def forward(self, inputs):
        return self.gen_fn(inputs)