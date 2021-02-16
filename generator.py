import six
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Bidirectional, \
    GRU, RNN, Layer, LSTM, LSTMCell, Dense, Flatten, Conv1D, Conv2D, \
        Conv2DTranspose, Reshape, Cropping1D, Lambda, Multiply, LeakyReLU
from tensorflow.keras import Model

from helper import *
from hilbert import *
from global_constants import *

def prep_hilb_for_net(t):
    amp = tf.squeeze(t[0])
    phase = tf.squeeze(t[1])
    #print("------------------------------------ First 10 samples (amp, phase):")
    #for i in range(10):
    #    print(float(amp[i].numpy()), " || ", float(phase[i].numpy()))
    z = np.dstack((amp, phase))
    z = tf.convert_to_tensor(z)
    hilb = tf.reshape(z, (N_BATCHES, N_TIMESTEPS, 2*N_UNITS//N_BATCHES))
    return hilb
def prep_hilb_for_dis(t):
    t = tf.squeeze(t)
    hilb = tf.reshape(t, (N_BATCHES, 1, TARGET_LEN_OVERRIDE//N_BATCHES))
    return hilb
def hilb_post_net(t):
    hilb = tf.reshape(t, (2, TARGET_LEN_OVERRIDE))
    amp = tf.squeeze(t[0])
    phase = tf.squeeze(t[1])
    #print("------------------------------------ First 10 samples (amp, phase):")
    #for i in range(10):
    #    print(float(amp[i].numpy()), " || ", float(phase[i].numpy()))
    return hilb

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Hilbert_Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Hilbert_Generator, self).__init__(**kwargs)
        self.state = None

        self.hilb_rnns = []
        for l in range(N_LAYERS):
            layer_f = GRU(N_UNITS*2, return_state=True, stateful=True, time_major=True, go_backwards=False, name="f_hilb_rnn_"+str(l))
            layer_b = GRU(N_UNITS*2, return_state=True, stateful=True, time_major=True, go_backwards=True, name="b_hilb_rnn_"+str(l))
            self.hilb_rnns.append(Bidirectional(layer_f, backward_layer=layer_b, merge_mode='ave'))
        
        self.optimizer = tf.keras.optimizers.Adam(GENERATOR_LR, 0.5)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(MODEL_DIR, "gen_ckpts"), max_to_keep=3)

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            print("Generator Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Generator Initializing from scratch.")

   
    def loss(self, op):
        return cross_entropy(tf.ones_like(op), op)
    
    def gen_fn(self, inputs, mode, weight_decay=2.5e-5):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        res, self.state = self.call(inputs, self.state, return_state=True, training=is_training)

        return res
    
    def call(self, inputs, state=None, return_state=False, training=False):
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
        hilb = prep_hilb_for_net(hilb)
        hilb.set_shape((N_BATCHES, N_TIMESTEPS, 2*N_UNITS//N_BATCHES))

        def prep_for_rnn(a, p):
            a = tf.expand_dims(a, axis=0)
            p = tf.expand_dims(p, axis=0)
            return a, p

        if state is None:
            f_state1 = self.hilb_rnns[0].forward_layer.get_initial_state(hilb)
            b_state1 = self.hilb_rnns[0].backward_layer.get_initial_state(hilb)
        else:
            f_state1, b_state1 = state
        f_state1 = f_state1[0]
        b_state1 = b_state1[0]

        print("|} Ready for launch! Going to the net now, wheeee!")

        hilb, f_state1, b_state1 = self.hilb_rnns[0](hilb, initial_state=(f_state1, b_state1), training=training)
        for i in range(N_LAYERS-1):
            hilb = tf.expand_dims(hilb, axis=0)
            hilb, f_state1, b_state1 = self.hilb_rnns[i+1](hilb, initial_state=(f_state1, b_state1), training=training)

        print("|} Whew... Made it out of the net alive!")

        hilb = hilb_post_net(hilb)

        return hilb, (f_state1, b_state1)