import six
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Bidirectional, \
    GRU, RNN, Layer, LSTM, LSTMCell, Dense, Flatten, Conv1D, Conv2D, \
        Conv1DTranspose, Reshape, Cropping1D, Lambda, Multiply, LeakyReLU
from tensorflow.keras import Model

from helper import *
from hilbert import *
from global_constants import *

def prep_hilb_for_rnn(t):
    amp = tf.squeeze(t[0])
    phase = tf.squeeze(t[1])
    z = np.dstack((amp, phase))
    z = tf.convert_to_tensor(z)
    hilb = tf.reshape(z, (N_BATCHES, N_TIMESTEPS, 2*N_UNITS//N_BATCHES))
    return hilb

def prep_hilb_for_deconv(t):
    z = tf.squeeze(t)
    z = K.flatten(z)
    hilb = tf.reshape(z, (N_BATCHES, N_TIMESTEPS, 2*N_UNITS//N_BATCHES))
    return hilb

def prep_hilb_for_dense(t):
    amp = tf.squeeze(t[0])
    phase = tf.squeeze(t[1])
    z = np.dstack((amp, phase))
    z = tf.convert_to_tensor(z)
    z = tf.expand_dims(z, axis=0)
    hilb = tf.reshape(z, (N_BATCHES, N_TIMESTEPS, 2*N_UNITS//N_BATCHES))
    return hilb

def prep_hilb_for_dis(t):
    t = tf.squeeze(t)
    hilb = tf.reshape(t, (N_BATCHES, 1, TARGET_LEN_OVERRIDE//N_BATCHES))
    return hilb

def flatten_hilb(t):
    tmp = K.flatten(t)
    amp = tmp[::2]
    phase = tmp[1::2]

    return hilb_tensor(amp, phase)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Hilbert_Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Hilbert_Generator, self).__init__(**kwargs)
        self.rnn_state = None

        self.N_DECONV_FILTERS = (TARGET_LEN_OVERRIDE // (N_DECONV_LAYERS+1))

        if GEN_MODE == 1:
            self.hilb_rnns = []
            for l in range(N_LAYERS):
                layer_f = GRU(N_UNITS*2, return_state=True, stateful=True, time_major=True, go_backwards=False, name="f_hilb_rnn_"+str(l))
                layer_b = GRU(N_UNITS*2, return_state=True, stateful=True, time_major=True, go_backwards=True, name="b_hilb_rnn_"+str(l))
                self.hilb_rnns.append(Bidirectional(layer_f, backward_layer=layer_b, merge_mode='ave'))
        elif GEN_MODE == 0:
            self.hilb_denses = []
            self.hilb_deconv = []
            self.hilb_denses.append(Dense(N_UNITS//N_DENSE_LAYERS, activation=tf.keras.activations.tanh, input_shape=[TARGET_LEN_OVERRIDE]))
            for _ in range(N_DENSE_LAYERS-1):
                self.hilb_denses.append(Dense(N_UNITS//2, activation=tf.keras.activations.tanh))
            for i in range(1, N_DECONV_LAYERS):
                n_filts = i*self.N_DECONV_FILTERS
                v_print("Creating deconvolution layer with", n_filts, "filters.")
                self.hilb_deconv.append(Conv1DTranspose(n_filts, kernel_size=1, strides=1, padding='same'))
            v_print("Creating deconvolution layer with", TARGET_LEN_OVERRIDE // N_TIMESTEPS, "filters.")
            self.hilb_deconv.append(Conv1DTranspose(TARGET_LEN_OVERRIDE // N_TIMESTEPS, kernel_size=1, strides=1, padding='same'))
            #self.hilb_denses.append(Dense(TARGET_LEN_OVERRIDE, activation=tf.keras.activations.tanh))
            
        self.optimizer = tf.keras.optimizers.Adam(GENERATOR_LR, 0.5)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(MODEL_DIR, "gen_ckpts"), max_to_keep=1)

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            print("Generator Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Generator Initializing from scratch.")
   
    def loss(self, op):
        return cross_entropy(tf.ones_like(op), op)
    
    def gen_rnn(self, inputs, training=False):
        assert(GEN_MODE == 1)

        x = my_hilbert(inputs)
        hilb = hilb_tensor(x[0], x[1])
        amp, phase = hilb[0], hilb[1]

        amp = tf.cast(amp, tf.float32)
        phase = tf.cast(phase, tf.float32)
        amp = tf.expand_dims(amp, axis=0)
        phase = tf.expand_dims(phase, axis=0)
        amp = tf.expand_dims(amp, axis=0)
        phase = tf.expand_dims(phase, axis=0)
        
        amp_dim1 = amp.shape[0]
        amp_dim2 = amp.shape[1]
        amp_dim3 = amp.shape[2]
        phase_dim1 = phase.shape[0]
        phase_dim2 = phase.shape[1]
        phase_dim3 = phase.shape[2]

        hilb = hilb_tensor(amp, phase)
        hilb = prep_hilb_for_rnn(hilb)
        hilb.set_shape((N_BATCHES, N_TIMESTEPS, 2*N_UNITS//N_BATCHES))

        if self.rnn_state is None:
            f_state1 = self.hilb_rnns[0].forward_layer.get_initial_state(hilb)
            b_state1 = self.hilb_rnns[0].backward_layer.get_initial_state(hilb)
        else:
            f_state1, b_state1 = self.rnn_state
        f_state1 = f_state1[0]
        b_state1 = b_state1[0]

        v_print("|} Ready for launch! Going to the net now, wheeee!")

        hilb, f_state1, b_state1 = self.hilb_rnns[0](hilb, initial_state=(f_state1, b_state1), training=training)
        for i in range(N_LAYERS-1):
            hilb = tf.expand_dims(hilb, axis=0)
            hilb, f_state1, b_state1 = self.hilb_rnns[i+1](hilb, initial_state=(f_state1, b_state1), training=training)

        v_print("|} Whew... Made it out of the net alive!")

        self.rnn_state = (f_state1, b_state1)
        hilb = flatten_hilb(hilb)
        return hilb

    def gen_dense(self, inputs, training=False):
        assert(GEN_MODE == 0)
        x = my_hilbert(inputs)
        hilb = hilb_tensor(x[0], x[1])
        
        hilb = prep_hilb_for_dense(hilb)
        v_print("|} Ready for launch! Going to the net now, wheeee!")
        for i in range(N_DENSE_LAYERS):
            hilb = self.hilb_denses[i](hilb)
        hilb = prep_hilb_for_deconv(hilb)
        v_print("|} Deconvolving...")
        for i in range(N_DECONV_LAYERS):
            hilb = self.hilb_deconv[i](hilb)
        v_print("|} Whew... Made it out of the net alive!")
        
        hilb = tf.squeeze(hilb)

        hilb = flatten_hilb(hilb)
        return hilb

    def gen_fn(self, inputs, training=False):
        return self.gen_dense(inputs, training)
    
    def call(self, inputs, training=False):
        return self.gen_fn(inputs, training)