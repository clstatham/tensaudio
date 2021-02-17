import collections
import os
import time
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import six
import soundfile
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras

from discriminator import DPAM_Discriminator
from generator import TA_Generator, prep_hilb_for_dis
from global_constants import *
from helper import *
from hilbert import *
from network_model import *

print(tf.version.VERSION)

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.run_functions_eagerly(True)
K.clear_session()
tf.compat.v1.reset_default_graph() 

np.random.seed(int(round(time.time())))

total_amps = []
total_phases = []

total_gen_losses = []
total_dis_losses = []

def record_amp_phase(amp, phase):
    # total_amps.append(amp)
    # total_phases.append(phase)
    pass
def plot_metrics(i, show=False):
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join("plots", timestamp)
    # fig1 = plt.figure()
    # fig1.suptitle("Amplitude Envelopes")
    # ax1 = fig1.add_subplot(1, 1, 1, projection='3d', label="Amplitude Envelopes", )
    # X, Y = np.meshgrid(len(total_amps), len(total_amps[0]))
    # _total_amps = np.ndarray([len(total_amps), len(total_amps[0])])
    # _total_amps[:][:] = total_amps[:][:]
    # ax1.plot_wireframe(X, Y, _total_amps, rstride=1, cstride=1)

    # fig2 = plt.figure()
    # fig2.suptitle("Intantaneous Phases")
    # ax2 = fig2.add_subplot(1, 1, 1, projection='3d', label="Intantaneous Phases")
    # X, Y = np.meshgrid(len(total_phases), len(total_phases[0]))
    # _total_phases = np.ndarray([len(total_phases), len(total_phases[0])])
    # _total_phases[:][:] = total_phases[:][:]
    # ax2.plot_wireframe(X, Y, _total_phases, rstride=1, cstride=1)

    fig3 = plt.figure()
    fig3.suptitle("Gen Losses")
    ax3 = fig3.add_subplot(1, 1, 1, label="Gen Losses")
    ax3.plot(range(len(total_gen_losses)), total_gen_losses)
    
    fig4 = plt.figure()
    fig4.suptitle("Dis Losses")
    ax4 = fig4.add_subplot(1, 1, 1, label="Dis Losses")
    ax4.plot(range(len(total_dis_losses)), total_dis_losses)

    os.mkdir(dirname)

    # filename1 = str('AMP_'+timestamp+'_'+str(i)+'.png')
    # filename2 = str('PHASE_'+timestamp+'_'+str(i)+'.png')
    filename3 = str('GEN_'+timestamp+'_'+str(i)+'.png')
    filename4 = str('DIS_'+timestamp+'_'+str(i)+'.png')
    # fig1.savefig(os.path.join(dirname, filename1))
    # fig2.savefig(os.path.join(dirname, filename2))
    fig3.savefig(os.path.join(dirname, filename3))
    fig4.savefig(os.path.join(dirname, filename4))

    # if show:
    #     fig1.show()
    #     fig2.show()
    #     while True:
    #         try:
    #             time.sleep(1)
    #         except KeyboardInterrupt:
    #             break

def open_truncate_pad(name):
    ret = []
    for filename in os.listdir(os.path.join(os.getcwd(), RESOURCES_DIR, name)):
        f = soundfile.SoundFile(os.path.join(os.getcwd(), RESOURCES_DIR, name, filename), "r")
        ret.append((f.read(), f.samplerate))
    return ret

global frick
def iterate_and_resample(files):
    arr = []
    for i in range(len(files)):
        fn = str(i)
        frick = files[0][0]
        if len(frick.shape) > 1 and frick.shape[1] != 1:
            frick = np.transpose(frick)[0]
        frick = np.asfortranarray(frick)
        a = librosa.resample(frick, files[i][1], SAMPLE_RATE)

        arr.append(a.tolist())
    return arr


def select_input(idx):
    x = tf.cast(INPUTS[idx], tf.float32)
    if x.shape[0] < TARGET_LEN_OVERRIDE:
        x = K.concatenate((x, [0]*(TARGET_LEN_OVERRIDE-x.shape[0])))

    x = x[:TARGET_LEN_OVERRIDE]
    return normalize_audio(x)

def get_example(idx):
    x = tf.cast(EXAMPLES[idx], tf.float32)
    if x.shape[0] < TARGET_LEN_OVERRIDE:
        x = K.concatenate((x, [0]*(TARGET_LEN_OVERRIDE-x.shape[0])))

    x = x[:TARGET_LEN_OVERRIDE]
    return normalize_audio(x)
def get_random_example():
    y = get_example(np.random.randint(0, len(EXAMPLES)))
    return y

def get_example_result(idx):
    x = tf.cast(EXAMPLE_RESULTS[idx], tf.float32)
    if x.shape[0] < TARGET_LEN_OVERRIDE:
        x = K.concatenate((x, [0]*(TARGET_LEN_OVERRIDE-x.shape[0])))

    x = x[:TARGET_LEN_OVERRIDE]
    return normalize_audio(x)
def get_random_example_result():
    y = get_example_result(np.random.randint(0, len(EXAMPLE_RESULTS)))
    return y

# a, p = my_hilbert(select_input(0))
# fig0 = plt.figure()
# plt.plot(range(len(select_input(0))), select_input(0))
# plt.show()
# fig1 = plt.figure()
# plt.plot(range(len(a)), a, color='red')
# plt.show()
# fig2 = plt.figure()
# plt.plot(range(len(p)), p, color='red')
# plt.show()
# y = inverse_hilbert(a, p)
# fig4 = plt.figure()
# plt.plot(range(len(y)), y)
# plt.show()
# scaled1 = np.int16(y/np.max(np.abs(y)) * 32767)
# soundfile.write('test.wav', scaled1, SAMPLE_RATE, SUBTYPE)



class OneStep():
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def generate_one_step(self, inputs):
        predicted_logits = self.generator(inputs, training=False)

        return predicted_logits

@tf.function
def create_input(i, dirname):
        x = get_random_example()
        y = get_random_example_result()

        y = spectral_convolution(x, y)
        
        # if i % SAVE_EVERY_ITERS == 0 and dirname is not None:
        #     print("Writing convolved result to disk.")
        #     scaled1 = np.int16(y/np.max(np.abs(y)) * 32767)
        #     soundfile.write(dirname+'/example_result_'+str(i)+'_'+str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))+'.wav', scaled1, SAMPLE_RATE, SUBTYPE)
        return x, y

gen_tape = tf.GradientTape(persistent=False)
dis_tape = tf.GradientTape(persistent=False)

@tf.function
def train_on_random(i, dirname):
    x, y = create_input(i, dirname)
    _, z = create_input(i, None)
    noise = tf.random.normal([TARGET_LEN_OVERRIDE])
    v_print("Passing training data to models.")
    begin_time = time.time()
    with gen_tape, dis_tape:
        v_print("| Generating...")
        g = gen.gen_fn(x, training=True)
        record_amp_phase(g[0], g[1])
        v_print("| Discriminating...")
        #h = invert_hilb_tensor(g)
        real_o = dis(z, y)
        fake_o = dis(z, g)
        v_print("| Calculating Loss...")
        gen_loss = gen.loss(fake_o)
        dis_loss = dis.loss(real_o, fake_o)[0]
    total_gen_losses.append(gen_loss)
    total_dis_losses.append(dis_loss)
    v_print("| Applying gradients...")
    gen_grads = gen_tape.gradient(gen_loss, gen.trainable_weights)
    gen.optimizer.apply_gradients(zip(gen_grads, gen.trainable_weights))
    dis_grads = dis_tape.gradient(dis_loss, dis.trainable_weights)
    dis.optimizer.apply_gradients(zip(dis_grads, dis.trainable_weights))
    print("|] Gen Loss:", float(gen_loss))
    print("|] Dis Loss:", float(dis_loss))
    time_diff = time.time() - begin_time
    print("Models successfully trained in", str(round(time_diff, ndigits=2)), "seconds.")
    if i % SAVE_MODEL_EVERY_ITERS == 0:
        print("Saving checkpoints for models...")
        gen.manager.save(i)
        dis.manager.save(i)
    
    return g


def train_until_interrupt():
    states = None
    i = 0
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join("training", "training_"+timestamp)
    os.mkdir(dirname)
    print("="*80)
    print("MODEL TRAINING BEGINS AT", timestamp)
    print("="*80)
    print()
    while True:
        try:
            v_print("*"*40)
            print("Initiating iteration #", i)
            train_on_random(i, dirname)
            if i % SAVE_EVERY_ITERS == 0:
                print("Generating progress update...")
                begin_time = time.time()
                plot_metrics(i, show=False)
                out1 = onestep.generate_one_step(select_input(0))
                scaled1 = np.int16(normalize_audio(out1) * 32767)
                soundfile.write(dirname+'/progress'+str(i)+"_"+str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))+'.wav', scaled1, SAMPLE_RATE, SUBTYPE)
                time_diff = time.time() - begin_time
                print("Progress update generated in", str(round(time_diff, ndigits=2)), "seconds.")
            i += 1
        except KeyboardInterrupt:
            break
    print("Saving weights to disk...")
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join(MODEL_DIR, "gen_weights", timestamp)
    os.mkdir(dirname)
    gen.save_weights(os.path.join(dirname, "gen_weights"))
    dirname = os.path.join(MODEL_DIR, "dis_weights", timestamp)
    os.mkdir(dirname)
    dis.save_weights(os.path.join(dirname, "dis_weights"))
    print()
    print("="*80)
    print("MODEL TRAINING FINISHED AT", timestamp)
    print("="*80)
    return i

v_print("Opening examples...")
EXAMPLE_FILES = open_truncate_pad(EXAMPLES_DIR)
v_print("Opening example results...")
EXAMPLE_RESULT_FILES = open_truncate_pad(EXAMPLE_RESULTS_DIR)
v_print("Opening inputs...")
INPUT_FILES = open_truncate_pad(INPUTS_DIR)
v_print("We have", len(INPUT_FILES), "inputs in the folder.")

print("Resampling, stand by...")
EXAMPLES = iterate_and_resample(EXAMPLE_FILES)
INPUTS = iterate_and_resample(INPUT_FILES)
EXAMPLE_RESULTS = iterate_and_resample(EXAMPLE_RESULT_FILES)
v_print("Created", len(EXAMPLES), "Example Arrays and", len(EXAMPLE_RESULTS), "Example Result Arrays.")

gen = TA_Generator()
dis = DPAM_Discriminator()
gen.compile(optimizer=gen.optimizer, loss=gen.loss)
dis.compile(optimizer=dis.optimizer, loss=dis.loss)

onestep = OneStep(gen)

if __name__ == "__main__":
    i = train_until_interrupt()

    print("Generating...")
    inp = select_input(0)
    data = onestep.generate_one_step(inp)
    print("Done!")

    scaled1 = np.int16(normalize_audio(data) * 32767)
    soundfile.write('out1.wav', scaled1, SAMPLE_RATE, SUBTYPE)

    plot_metrics(i, show=True)
