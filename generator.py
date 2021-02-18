import six
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from helper import *
from hilbert import *
from global_constants import *

def prep_noise_for_pre_dense(t):
    return torch.reshape(t, (N_BATCHES, N_TIMESTEPS, 1))

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
            self.audio_rnns.append(nn.LSTM(SAMPLES_PER_BATCH, return_state=True, stateful=True, time_major=False, go_backwards=False, name="f_audio_rnn_0"))
            for i in range(N_RNN_LAYERS-1):
                layer_f = LSTM(SAMPLES_PER_BATCH, return_state=True, stateful=True, time_major=False, go_backwards=False, name="f_audio_rnn_"+str(i+1))
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
        self.pre_dense = []
        self.denses = []
        self.post_dense = []
        v_print("*-"*39 + "*")
        if hilb_mode:
            # need double of everything for double the data!
            _TOTAL_SAMPLES = 2 * TOTAL_SAMPLES
            _N_BATCHES = 2 * N_BATCHES
            _SAMPLES_PER_BATCH = 2 * SAMPLES_PER_BATCH
            _N_UNITS = N_UNITS
            _N_POST_DENSE_BATCHES = _N_BATCHES
            _N_PRE_DENSE_FILTERS = 2 * N_PRE_DENSE_FILTERS
            _N_POST_DENSE_FILTERS = N_POST_DENSE_FILTERS // 2
        else:
            _N_BATCHES, _SAMPLES_PER_BATCH, _N_UNITS, _N_POST_DENSE_BATCHES, _N_PRE_DENSE_FILTERS, _N_POST_DENSE_FILTERS = \
                N_BATCHES, SAMPLES_PER_BATCH, N_UNITS, N_POST_DENSE_BATCHES, N_PRE_DENSE_FILTERS, N_POST_DENSE_FILTERS
        
        if USE_REAL_AUDIO:
            v_print("Creating convolution layer", N_TIMESTEPS, ">", _SAMPLES_PER_BATCH)
            #self.pre_dense.append(nn.Tanh())
            self.pre_dense.append(nn.Conv1d(N_TIMESTEPS, _SAMPLES_PER_BATCH, kernel_size=KERNEL_SIZE, stride=1))
            for i in range(N_PRE_DENSE_LAYERS-1):
                v_print("Creating convolution layer", _SAMPLES_PER_BATCH, ">", _SAMPLES_PER_BATCH)
                #self.pre_dense.append(nn.Tanh())
                self.pre_dense.append(nn.Conv1d(_SAMPLES_PER_BATCH, _SAMPLES_PER_BATCH, kernel_size=KERNEL_SIZE, stride=1))
        else:
            # must expand a smaller vector into something we can use!
            v_print("Creating deconvolution layer", N_TIMESTEPS, ">", _SAMPLES_PER_BATCH)
            #self.pre_dense.append(nn.Tanh())
            self.pre_dense.append(nn.ConvTranspose1d(N_TIMESTEPS, _SAMPLES_PER_BATCH, kernel_size=KERNEL_SIZE, stride=1))
            # for i in range(1,N_PRE_DENSE_LAYERS):
            #     n_filts = i*_N_PRE_DENSE_FILTERS
            #     v_print("Creating deconvolution layer", _SAMPLES_PER_BATCH, ">", _SAMPLES_PER_BATCH)
            #     self.pre_dense.append(nn.ConvTranspose1d(_SAMPLES_PER_BATCH, _SAMPLES_PER_BATCH, kernel_size=KERNEL_SIZE, stride=1))
        
        v_print("Creating convolution layer", _SAMPLES_PER_BATCH, ">", _SAMPLES_PER_BATCH*2)
        #self.pre_dense.append(nn.Tanh())
        self.pre_dense.append(nn.Conv1d(_SAMPLES_PER_BATCH, _SAMPLES_PER_BATCH*2, kernel_size=KERNEL_SIZE, stride=1))

        # for i in range(N_DENSE_LAYERS-1):
        #     v_print("Creating linear layer", _SAMPLES_PER_BATCH, ">", _TOTAL_SAMPLES)
        #     #self.denses.append(nn.Tanh())
        #     self.denses.append(nn.Linear(_SAMPLES_PER_BATCH, _TOTAL_SAMPLES))
        # v_print("Creating linear layer", _N_UNITS, ">", _N_UNITS)
        # #self.denses.append(nn.Tanh())
        # self.denses.append(nn.Linear(640, 640))

        # for i in range(1,N_POST_DENSE_LAYERS):
        #     n_filts = i*N_POST_DENSE_FILTERS
        #     v_print("Creating deconvolution layer", _SAMPLES_PER_BATCH, ">", N_TIMESTEPS)
        #     self.post_dense.append(nn.ConvTranspose1d(_SAMPLES_PER_BATCH, N_TIMESTEPS, kernel_size=KERNEL_SIZE, stride=1))
        v_print("Creating deconvolution layer", N_TIMESTEPS, ">", N_TIMESTEPS)
        self.post_dense.append(nn.Conv1d(N_TIMESTEPS, N_TIMESTEPS, kernel_size=KERNEL_SIZE, stride=1))
        v_print("*-"*39 + "*")

        #for i in range(len(self.pre_dense)):
        #    self.register_parameter("pre_dense_"+str(i), self.pre_dense[i])
        #for i in range(len(self.denses)):
        #    self.register_parameter("dense_"+str(i), self.denses[i])
        #for i in range(len(self.post_dense)):
        #    self.register_parameter("post_dense_"+str(i), self.post_dense[i])
        self.pre_dense = nn.ModuleList(self.pre_dense).cuda()
        self.denses = nn.ModuleList(self.denses).cuda()
        self.post_dense = nn.ModuleList(self.post_dense).cuda()

    def gen_rnn(self, inputs, training=False):
        assert(GEN_MODE == 1)

        audio = inputs

        audio = prep_audio_for_batch_operation(audio)

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
            audio = prep_audio_for_batch_operation(audio)
            #audio = tf.expand_dims(audio, axis=0)
            audio, f_state1, f_state2 = self.audio_rnns[i+1](audio, initial_state=(f_state1, f_state2), training=training)
            v_print("|}} Layer", i+2, "done.")

        v_print("|} Whew... Made it out of the net alive!")

        #self.rnn_state = (f_state1, f_state2)
        audio = K.flatten(audio)
        return audio

    def run_dense_net(self, data, hilb_mode=False):
        for i in range(len(self.pre_dense)):
            data = self.pre_dense[i](data.float())
            v_print("|}} Pre-Dense layer", i+1, "done.")
        #data = data[:,-1,:]
        # for i in range(len(self.denses)):
        #     data = data.flatten()
        #     v_print("|}} data.shape:", data.shape)
        #     v_print("|}} ", self.denses[i].in_features)
        #     data = self.denses[i](data)
        #     v_print("|}} Dense layer", i+1, "done.")
        v_print("|}} data.shape:", data.shape)
        data = prep_hilb_for_batch_operation(data, (32*N_BATCHES),N_TIMESTEPS,KERNEL_SIZE)
        for i in range(len(self.post_dense)):
            v_print("|}} data.shape:", data.shape)
            data = self.post_dense[i](data)
            v_print("|}} Post-Dense layer", i+1, "done.")
        return data

    def gen_dense_hilb(self, hilb, training=False):
        assert (GEN_MODE == 2)
        hilb = torch.tensor(stack_hilb(hilb)).cuda()
        hilb = prep_hilb_for_batch_operation(hilb, (2*N_BATCHES),N_TIMESTEPS,N_UNITS//KERNEL_SIZE)
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
        
        if USE_REAL_AUDIO:
            audio = prep_audio_for_batch_operation(audio)
        else:
            audio = prep_noise_for_pre_dense(audio)
        v_print("|} Ready for launch! Going to the net now, wheeee!")
        audio = self.run_dense_net(audio)
        v_print("|} Whew... Made it out of the net alive!")
        
        audio = torch.squeeze(audio)
        audio = audio[:, -1, :] # the last timestep

        audio = torch.flatten(audio)
        assert(audio.shape[0] == TOTAL_SAMPLES)
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