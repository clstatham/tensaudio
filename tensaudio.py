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
import torch
import torchaudio

from discriminator import DPAM_Discriminator
from generator import TA_Generator
from global_constants import *
from helper import *
from hilbert import *

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
    dirname = os.path.join(PLOTS_DIR, timestamp)
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
        frick = files[i][0]
        if len(frick.shape) > 1 and frick.shape[1] != 1:
            frick = np.transpose(frick)[0]
        frick = np.asfortranarray(frick)
        a = librosa.resample(frick, files[i][1], SAMPLE_RATE)

        arr.append(a.tolist())
    return arr


def select_input(idx):
    x = torch.tensor(INPUTS[idx]).cuda()
    if len(x) < TOTAL_SAMPLES:
        x = torch.cat((x, torch.tensor([0]*(TOTAL_SAMPLES-len(x))).cuda()))

    x = x[SLICE_START:TOTAL_SAMPLES]
    return normalize_audio(x)

def get_example(idx):
    x = torch.tensor(EXAMPLES[idx]).cuda()
    if len(x) < TOTAL_SAMPLES:
        x = torch.cat((x, torch.tensor([0]*(TOTAL_SAMPLES-len(x))).cuda()))

    x = x[SLICE_START:TOTAL_SAMPLES]
    return normalize_audio(x)
def get_random_example():
    return get_example(np.random.randint(len(EXAMPLES)))

def get_example_result(idx):
    x = torch.tensor(EXAMPLE_RESULTS[idx]).cuda()
    if len(x) < TOTAL_SAMPLES:
        x = torch.cat((x, torch.tensor([0]*(TOTAL_SAMPLES-len(x))).cuda()))

    x = x[SLICE_START:TOTAL_SAMPLES]
    return normalize_audio(x)
def get_random_example_result():
    return get_example_result(np.random.randint(len(EXAMPLE_RESULTS)))

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

def create_inputs():
        x = get_random_example()
        y = get_random_example_result()

        if INPUT_MODE == 'conv':
            y = torch.from_numpy(spectral_convolution(x, y))
        return x, y

def generate_input_noise():
    return torch.rand([TOTAL_SAMPLES])

gen_loss, dis_loss = None, None

def run_models(x, y, z):
    global gen_loss, dis_loss
    v_print("| Generating...")
    g = gen.forward(x, training=True)
    v_print("| g.shape:", g.shape)
    record_amp_phase(g[0], g[1])
    v_print("| Discriminating...")
    real_o = torch.flatten(dis.forward(z, y))
    fake_o = torch.flatten(dis.forward(z, g))
    v_print("| Calculating Loss...")
    gen_loss = gen.criterion(fake_o)
    dis_loss = dis.criterion(real_o, fake_o)
    gen_loss.backward(retain_graph=True)
    dis_loss.backward()
    print("|] Fake Verdict:\t", round(float(fake_o), 4))
    print("|] Real Verdict:\t", round(float(real_o), 4))
    return g

def train_on_random(i, dirname):
    x, y = create_inputs()
    _, z = create_inputs()
    if not USE_REAL_AUDIO:
        x = generate_input_noise()
    if SAVE_EVERY_ITERS > 0 and i % SAVE_EVERY_ITERS == 0 and dirname is not None:
        print("Writing training data to disk.")
        timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
        scaled1 = np.int16(normalize_audio(x.cpu()) * 32767)
        scaled2 = np.int16(normalize_audio(y.cpu()) * 32767)
        scaled3 = np.int16(normalize_audio(z.cpu()) * 32767)
        soundfile.write(dirname+'/iter'+str(i)+'_x_'+timestamp+'.wav', scaled1, SAMPLE_RATE, SUBTYPE)
        soundfile.write(dirname+'/iter'+str(i)+'_y_'+timestamp+'.wav', scaled2, SAMPLE_RATE, SUBTYPE)
        soundfile.write(dirname+'/iter'+str(i)+'_z_'+timestamp+'.wav', scaled3, SAMPLE_RATE, SUBTYPE)
    v_print("Passing training data to models.")
    begin_time = time.time()
    gen_optim.zero_grad()
    dis_optim.zero_grad()
    g = run_models(x, y, z)
    total_gen_losses.append(gen_loss)
    total_dis_losses.append(dis_loss)
    print("|] Gen Loss:\t\t", round(float(gen_loss), 4))
    print("|] Dis Loss:\t\t", round(float(dis_loss), 4))
    v_print("| Applying gradients...")
    gen_optim.step()
    dis_optim.step()

    time_diff = time.time() - begin_time
    print("Models successfully trained in", str(round(time_diff, ndigits=2)), "seconds.")
    # if SAVE_MODEL_EVERY_ITERS > 0 and i % SAVE_MODEL_EVERY_ITERS == 0:
    #     print("Saving checkpoints for models...")
    #     torch.save(gen, os.path.join(MODEL_DIR, "gen_ckpts"))
    #     torch.save(dis, os.path.join(MODEL_DIR, "dis_ckpts"))
    
    return g


def train_until_interrupt():
    states = None
    i = 0
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join(TRAINING_DIR, "training_"+timestamp)
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
            if SAVE_EVERY_ITERS > 0 and i % SAVE_EVERY_ITERS == 0:
                print("Generating progress update...")
                begin_time = time.time()
                plot_metrics(i, show=False)
                if USE_REAL_AUDIO:
                    out1 = onestep.generate_one_step(select_input(0))
                else:
                    out1 = onestep.generate_one_step(generate_input_noise())
                write_normalized_audio_to_disk(out1, dirname+'/progress'+str(i)+"_"+str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))+'.wav')
                time_diff = time.time() - begin_time
                print("Progress update generated in", str(round(time_diff, ndigits=2)), "seconds.")
            i += 1
        except KeyboardInterrupt:
            break
    # print("Saving weights to disk...")
    # timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    # dirname = os.path.join(MODEL_DIR, "gen_weights", timestamp)
    # os.mkdir(dirname)
    # gen.save_weights(os.path.join(dirname, "gen_weights"))
    # dirname = os.path.join(MODEL_DIR, "dis_weights", timestamp)
    # os.mkdir(dirname)
    # dis.save_weights(os.path.join(dirname, "dis_weights"))
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

gen = TA_Generator().cuda()
dis = DPAM_Discriminator().cuda()
gen_optim = torch.optim.SGD(gen.parameters(), lr=GENERATOR_LR, momentum=0.9)
dis_optim = torch.optim.SGD(dis.parameters(), lr=DISCRIMINATOR_LR, momentum=0.9)

onestep = OneStep(gen)

if __name__ == "__main__":
    i = train_until_interrupt()

    print("Generating...")
    if USE_REAL_AUDIO:
        data = onestep.generate_one_step(select_input(0))
    else:
        data = onestep.generate_one_step(generate_input_noise())
    print("Done!")

    write_normalized_audio_to_disk(data, 'out1.wav')

    plot_metrics(i)
