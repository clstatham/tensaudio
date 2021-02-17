import six
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, Bidirectional, \
    GRU, SimpleRNN, Layer, LSTM, LSTMCell, Dense, Flatten, Conv1D, Conv2D, \
        Conv1DTranspose, Reshape, Cropping1D, Lambda, Multiply, LeakyReLU
from tensorflow.keras import Model

from helper import *
from hilbert import *
from global_constants import *

@tf.function
def prep_hilb_for_rnn(t):
    #z = tf.Variable(np.dstack((tf.squeeze(t[0]), tf.squeeze(t[1]))), trainable=False)
    hilb = tf.convert_to_tensor(tf.reshape(t, (2*N_BATCHES, N_TIMESTEPS, N_UNITS//2)))
    return hilb

@tf.function
def prep_hilb_for_deconv(t):
    #z = tf.Variable(np.dstack((tf.squeeze(t[0]), tf.squeeze(t[1]))), trainable=False)
    hilb = tf.convert_to_tensor(tf.reshape(t, (2*N_BATCHES, N_TIMESTEPS, N_UNITS//2)))
    return hilb

@tf.function
def prep_hilb_for_dense(t):
    #z = tf.Variable(np.dstack((tf.squeeze(t[0]), tf.squeeze(t[1]))), trainable=False)
    hilb = tf.convert_to_tensor(tf.reshape(t, (2*N_BATCHES, N_TIMESTEPS, N_UNITS//2)))
    return hilb

@tf.function
def prep_hilb_for_dis(t):
    #z = tf.Variable(np.dstack((tf.squeeze(t[0]), tf.squeeze(t[1]))), trainable=False)
    hilb = tf.convert_to_tensor(tf.reshape(t, (N_BATCHES, 1, TARGET_LEN_OVERRIDE//N_BATCHES)))
    return hilb

@tf.function
def flatten_hilb(t):
    tmp = tf.Variable(K.flatten(t), trainable=False)
    return hilb_tensor(tmp[::2], tmp[1::2])
@tf.function
def flatten_audio(t):
    return K.flatten(t)

# foo = np.arange(0, N_BATCHES*2)
# print(foo)
# foo = prep_audio_for_batch_operation(foo)
# print(foo)
# foo = flatten_audio(foo)
# print(foo)

class TA_Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(TA_Generator, self).__init__(**kwargs)
        self.rnn_state = None

        if GEN_MODE == 1:
            self.audio_rnns = []
            self.audio_rnns.append(LSTM(TARGET_LEN_OVERRIDE//N_BATCHES, return_state=True, stateful=True, time_major=False, go_backwards=False, name="f_audio_rnn_0"))
            for i in range(N_RNN_LAYERS-1):
                layer_f = LSTM(TARGET_LEN_OVERRIDE//N_BATCHES, return_state=True, stateful=True, time_major=False, go_backwards=False, name="f_audio_rnn_"+str(i+1))
                #layer_b = SimpleRNN(N_UNITS*2, return_state=True, stateful=True, time_major=True, go_backwards=True, name="b_audio_rnn_"+str(l))
                #self.audio_rnns.append(Bidirectional(layer_f, backward_layer=layer_b, merge_mode='ave'))
                self.audio_rnns.append(layer_f)
        elif GEN_MODE == 0:
            self.audio_conv = []
            self.audio_denses = []
            self.audio_deconv = []
            print("*-"*39 + "*")
            v_print("Creating convolution layer with", SAMPLES_PER_BATCH, "filters.")
            self.audio_conv.append(Conv1D(SAMPLES_PER_BATCH, kernel_size=KERNEL_SIZE, activation=tf.keras.activations.tanh, input_shape=(N_BATCHES, N_TIMESTEPS, N_UNITS), name="audio_conv_0"))
            for i in range(N_CONV_LAYERS-1):
                v_print("Creating convolution layer with", SAMPLES_PER_BATCH, "filters.")
                self.audio_conv.append(Conv1D(SAMPLES_PER_BATCH, kernel_size=KERNEL_SIZE, strides=1, activation=tf.keras.activations.tanh, name="audio_conv_"+str(i)))
            for i in range(N_DENSE_LAYERS-1):
                v_print("Creating dense layer with", N_TIMESTEPS, "units.")
                self.audio_denses.append(Dense(N_TIMESTEPS, activation=tf.keras.activations.tanh, name="audio_dense_"+str(i)))
            v_print("Creating dense layer with", SAMPLES_PER_BATCH, "units.")
            self.audio_denses.append(Dense(SAMPLES_PER_BATCH, activation=tf.keras.activations.tanh, name="audio_dense_"+str(N_DENSE_LAYERS)))
            for i in range(1,N_DECONV_LAYERS):
                n_filts = i*N_DECONV_FILTERS
                v_print("Creating deconvolution layer with", n_filts, "filters.")
                self.audio_deconv.append(Conv1DTranspose(n_filts, kernel_size=KERNEL_SIZE, strides=1, padding='same', name="audio_conv1dt_"+str(i-1)))
            #TODO: figure out wtf this magic number 20 is
            v_print("Creating deconvolution layer with", N_UNITS*N_TIMESTEPS//20, "filters.")
            self.audio_deconv.append(Conv1DTranspose(N_UNITS*N_TIMESTEPS//20, kernel_size=KERNEL_SIZE, strides=1, padding='same', name="audio_conv1dt_"+str(N_DECONV_LAYERS-1)))
            print("*-"*39 + "*")
        
        self.optimizer = tf.keras.optimizers.SGD(GENERATOR_LR)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(MODEL_DIR, "gen_ckpts"), max_to_keep=1)

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            print("Generator Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Generator Initializing from scratch.")
   
    @tf.function
    def loss(self, op):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(op), op)
    
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

    def gen_dense(self, inputs, training=False):
        assert(GEN_MODE == 0)
        #inputs = tf.expand_dims(inputs, axis=0)
        #audio = my_audioert(inputs)
        #audio = audio_tensor(x[0], x[1])
        audio = inputs
        
        audio = prep_audio_for_batch_operation(audio)
        v_print("|} Ready for launch! Going to the net now, wheeee!")
        for i in range(N_CONV_LAYERS):
            audio = self.audio_conv[i](audio)
            v_print("|}} Convolution layer", i+1, "done.")
        for i in range(N_DENSE_LAYERS):
            audio = self.audio_denses[i](audio)
            v_print("|}} Dense layer", i+1, "done.")
        for i in range(N_DECONV_LAYERS):
            audio = self.audio_deconv[i](audio)
            v_print("|}} Deconvolution layer", i+1, "done.")
        v_print("|} Whew... Made it out of the net alive!")
        
        audio = tf.squeeze(audio)

        audio = flatten_audio(audio)
        return audio

    @tf.function
    def gen_fn(self, inputs, training=False):
        if GEN_MODE == 0:
            return self.gen_dense(inputs, training)
        elif GEN_MODE == 1:
            return self.gen_rnn(inputs, training)
    
    @tf.function
    def call(self, inputs, training=False):
        return self.gen_fn(inputs, training)