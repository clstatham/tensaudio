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

        self.loss = nn.L1Loss()

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

    def criterion(self, op):
        #return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(op), op)
        return self.loss(input=op, target=torch.tensor(0.).cuda())
    
    def create_dense_net(self, hilb_mode=False):
        self.preprocess_layers = []
        self.process_layers = []
        self.postprocess_layers = []
        self.n_hidden_layers = 64

        v_print("*-"*39 + "*")
        
        self.n_process_units = TOTAL_SAMPLES
        self.offset = N_PREPROCESS_LAYERS+3
        v_print("Creating convolution layer", N_CHANNELS, ">", self.offset+N_CHANNELS*(N_PREPROCESS_LAYERS))
        self.preprocess_layers.append(nn.Conv1d(N_CHANNELS, self.offset+N_CHANNELS*(N_PREPROCESS_LAYERS), kernel_size=KERNEL_SIZE, stride=1).cuda())
        n_inputs = 0
        n_outputs = 0
        ksz = KERNEL_SIZE
        stride = 0
        total_lost_dims = 0
        for i in range(N_PREPROCESS_LAYERS):
            n_inputs = 1+self.offset+N_CHANNELS*(N_PREPROCESS_LAYERS-i)
            n_outputs = self.offset+N_CHANNELS*(N_PREPROCESS_LAYERS-i)
            #self.n_process_units = n_outputs
            stride = ksz * (i+1)
            total_lost_dims += (n_outputs * ksz * stride)
            v_print("Creating convolution layer", n_inputs, ">", n_outputs, "ksz =", ksz, "stride =", stride)
            self.preprocess_layers.append(nn.Conv1d(n_inputs, n_outputs, kernel_size=ksz, stride=stride).cuda())
        # stride = ksz * N_PREPROCESS_LAYERS
        # v_print("Creating convolution layer", 1, ">", 1, "ksz =", ksz, "stride =", stride)
        # self.preprocess_layers.append(nn.Conv1d(1, 1, kernel_size=ksz, groups=1, stride=stride).cuda())

        self.final_preprocess_stride = stride

        self.n_process_units = self.n_process_units / total_lost_dims
        self.n_process_units = 2 * int(self.n_process_units**2 / N_TIMESTEPS_PER_KERNEL)
        for i in range(N_PROCESS_LAYERS):
            v_print("Creating Linear layer", self.n_process_units, ">", self.n_process_units)
            self.process_layers.append(nn.Linear(self.n_process_units, self.n_process_units).cuda())
        
        n_deconv_filters = 0
        deconv_kernel_size = KERNEL_SIZE
        for i in range(1, N_POSTPROCESS_LAYERS):
            n_deconv_filters = (1+N_CHANNELS)**(i+1)
            deconv_kernel_size = KERNEL_SIZE**i
            v_print("Creating deconvolution layer", (1+N_CHANNELS)**i, ">", n_deconv_filters)
            self.postprocess_layers.append(nn.ConvTranspose1d((1+N_CHANNELS)**i, n_deconv_filters, kernel_size=deconv_kernel_size, stride=1).cuda())
        v_print("Creating convolution layer", n_deconv_filters, ">", N_CHANNELS)
        self.postprocess_layers.append(nn.Conv1d(n_deconv_filters, N_CHANNELS, kernel_size=KERNEL_SIZE, stride=1).cuda())
        
        self.preprocess_layers = nn.ModuleList(self.preprocess_layers).cuda()
        self.process_layers = nn.ModuleList(self.process_layers).cuda()
        self.postprocess_layers = nn.ModuleList(self.postprocess_layers).cuda()

    def gen_rnn(self, inputs, training=False):
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

        v_print("|} Ready for launch! Going to the net now, wheeee!")

        audio, f_state1, f_state2 = self.audio_rnns[0](audio, initial_state=(f_state1, f_state2), training=training)
        v_print("|} Layer 1 done.")
        for i in range(N_RNN_LAYERS-1):
            audio = prep_data_for_batch_operation(audio)
            #audio = tf.expand_dims(audio, axis=0)
            audio, f_state1, f_state2 = self.audio_rnns[i+1](audio, initial_state=(f_state1, f_state2), training=training)
            v_print("|}} Layer", i+2, "done.")

        v_print("|} Whew... Made it out of the net alive!")

        #self.rnn_state = (f_state1, f_state2)
        audio = K.flatten(audio)
        return audio

    def run_dense_net(self, data, hilb_mode=False):
        total_delta = 0
        
        v_print("|}} Initial data.shape:", data.shape)
        
        data, delta = prep_data_for_batch_operation(data, N_BATCHES, N_CHANNELS, None, greedy=True)
        total_delta += delta
        data = self.preprocess_layers[0](data.float())
        v_print("|}} Pre-process layer 1 done.")
        v_print("|}} data.shape:", data.shape)
        for i in range(N_PREPROCESS_LAYERS-1):
            data, delta = prep_data_for_batch_operation(data, N_BATCHES, 2+self.offset+N_CHANNELS*(N_PREPROCESS_LAYERS-(i+1)), None, greedy=True)
            total_delta += delta
            data = self.preprocess_layers[i+1](data.float())
            v_print("|}} Pre-process layer", i+2, "done.")
            v_print("|}} data.shape:", data.shape)
        v_print("|}} Chopping data to final batch (", data.shape[0], ") and timestep (", data.shape[1], ")")
        data = data[-1:, -1, :]
        data, delta = prep_data_for_batch_operation(data, 1, 1, None, greedy=True)
        total_delta += delta
        v_print("|}} data.shape:", data.shape)
        #data = self.preprocess_layers[-1](data.float())
        #v_print("|}} Pre-process layer", N_PREPROCESS_LAYERS, "done.")
        #v_print("|}} data.shape:", data.shape)

        # remove any zero padding we've had to add during reshapes
        data = data.flatten()
        if total_delta > 0:
            v_print("|}} Removing delta:", total_delta//self.final_preprocess_stride)
            data = data[:-total_delta//self.final_preprocess_stride]
        total_delta = 0

        data, delta = prep_data_for_batch_operation(data, 1, self.n_process_units, None, greedy=True)
        total_delta += delta
        data = torch.squeeze(data)
        data = data.flatten()
        v_print("|}} data.shape going into process layers:", data.shape)
        for i in range(len(self.process_layers)):
            data = self.process_layers[i](data)
            data = data.flatten()
            v_print("|}} Process layer", i+1, "done.")
            v_print("|}} data.shape:", data.shape)

        for i in range(1, len(self.postprocess_layers)):
            data, delta = prep_data_for_batch_operation(data, N_BATCHES, (1+N_CHANNELS)**i, None, greedy=True)
            total_delta += delta
            data = self.postprocess_layers[i-1](data)
            v_print("|}} Post-process layer", i, "done.")
            v_print("|}} data.shape:", data.shape)
        data, delta= prep_data_for_batch_operation(data, N_BATCHES, (1+N_CHANNELS)**N_POSTPROCESS_LAYERS, None, greedy=True)
        total_delta += delta
        data = self.postprocess_layers[-1](data)
        v_print("|}} Post-process layer", N_POSTPROCESS_LAYERS, "done.")
        v_print("|}} data.shape:", data.shape)
        data = data.flatten()
        if total_delta > 0:
            v_print("|}} Removing delta:", total_delta)
            data = data[:-total_delta]
        total_delta = 0
        v_print("|}} Final data.shape:", data.shape)
        return data

    def gen_dense_hilb(self, hilb, training=False):
        assert (GEN_MODE == 2)
        hilb = torch.tensor(stack_hilb(hilb)).cuda()
        hilb = prep_data_for_batch_operation(hilb, N_BATCHES, N_CHANNELS, None)
        v_print("|} Ready for launch! Going to the net now, wheeee!")
        hilb = self.run_dense_net(hilb, hilb_mode=True)
        v_print("|} Whew... Made it out of the net alive!")
        #hilb = tf.squeeze(hilb)
        #hilb = hilb[-1, :, :]
        v_print("|}} hilb.shape:", hilb.shape)
        hilb = torch.flatten(hilb)
        hilb = hilb[::2], hilb[1::2]
        assert(len(hilb) == 2 and len(hilb[0] == TOTAL_SAMPLES))
        return hilb

    def gen_dense(self, inputs, training=False):
        assert(GEN_MODE in (2, 3))
        #inputs = tf.expand_dims(inputs, axis=0)
        #audio = my_audioert(inputs)
        #audio = audio_tensor(x[0], x[1])
        audio = inputs
        orig_shape = audio.shape
        
        audio = prep_data_for_batch_operation(audio, N_BATCHES, N_CHANNELS, None)
        v_print("|} Ready for launch! Going to the net now, wheeee!")
        audio = self.run_dense_net(audio, hilb_mode=False)
        v_print("|} Whew... Made it out of the net alive!")
        
        audio = torch.squeeze(audio)
        #audio = audio[:, -1, :] # the last

        audio = torch.flatten(audio)
        audio, error = ensure_size(audio, TOTAL_SAMPLES)
        if error > 0:
            v_print("|} Truncated output audio by", error, "samples.")
        else:
            v_print("|} Padded output audio by", -error, "samples.")
        return audio

    def gen_fn(self, inputs, training=False):
        if GEN_MODE == 0:
            amp, phase = my_hilbert(inputs)
            hilb = self.gen_rnn_hilb(torch.stack((amp, phase)).cuda(), training)
            return invert_hilb_tensor(hilb)
        if GEN_MODE == 1:
            return self.gen_rnn(inputs, training)
        if GEN_MODE == 2:
            amp, phase = my_hilbert(inputs)
            hilb = self.gen_dense_hilb(torch.stack((amp, phase)).cuda(), training)
            return invert_hilb_tensor(hilb)
        if GEN_MODE == 3:
            return self.gen_dense(inputs, training)
    
    def forward(self, inputs, training=False):
        return self.gen_fn(inputs, training)