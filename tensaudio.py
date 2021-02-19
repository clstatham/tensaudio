import collections
import os
import time
from datetime import datetime
import curses

import librosa
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.io.wavfile
import six
import soundfile
import torch
import torch.nn as nn

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
total_real_verdicts = []
total_fake_verdicts = []

def record_amp_phase(amp, phase):
    # total_amps.append(amp)
    # total_phases.append(phase)
    pass
def plot_metrics(i, show=False):
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join(PLOTS_DIR, timestamp)
    os.mkdir(dirname)

    filename1 = str('AMP_'+timestamp+'_'+str(i)+'.png')
    filename2 = str('PHASE_'+timestamp+'_'+str(i)+'.png')
    filename3 = str('LOSSES_'+timestamp+'_'+str(i)+'.png')

    # if len(total_amps) > 1:
    #     fig1 = plt.figure(figsize=(10,10))
    #     fig1.suptitle("Amplitude Envelopes "+ timestamp)
    #     ax1 = fig1.add_subplot(111, projection='3d')
    #     Y, X = np.meshgrid(range(len(total_amps)), range(len(total_amps[0])))
    #     Z = np.reshape(np.array(total_amps), X.shape)
    #     ax1.scatter(X, Y, Z, s=1, cmap=cm.coolwarm)
    #     fig1.savefig(os.path.join(dirname, filename1))

    # if len(total_phases) > 1:
    #     fig2 = plt.figure(figsize=(10,10))
    #     fig2.suptitle("Instantaneous Phases "+ timestamp)
    #     ax2 = fig2.add_subplot(111, projection='3d')
    #     Y, X = np.meshgrid(range(len(total_phases)), range(len(total_phases[0])))
    #     Z = np.reshape(np.array(total_phases), X.shape)
    #     ax2.scatter(X, Y, Z, s=1, cmap=cm.coolwarm)
    #     fig2.savefig(os.path.join(dirname, filename2))

    if len(total_gen_losses) + len(total_dis_losses) + len(total_real_verdicts) + len(total_fake_verdicts) > 0:
        fig3 = plt.figure(figsize=(10,10))
        fig3.suptitle("Gen/Dis Losses " + timestamp)
        plt.plot(range(len(total_gen_losses)), total_gen_losses, label="Gen Losses")
        plt.plot(range(len(total_dis_losses)), total_dis_losses, label="Dis Losses")
        plt.plot(range(len(total_real_verdicts)), total_real_verdicts, label="Real Verdicts")
        plt.plot(range(len(total_fake_verdicts)), total_fake_verdicts, label="Fake Verdicts")
        plt.legend()
        fig3.savefig(os.path.join(dirname, filename3))

    plt.close()

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
    if len(x) < TOTAL_SAMPLES_OUT:
        x = torch.cat((x, torch.tensor([0]*(TOTAL_SAMPLES_OUT-len(x))).cuda()))

    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_audio(x)

def get_example(idx):
    x = torch.tensor(EXAMPLES[idx]).cuda()
    if len(x) < TOTAL_SAMPLES_OUT:
        x = torch.cat((x, torch.tensor([0]*(TOTAL_SAMPLES_OUT-len(x))).cuda()))

    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_audio(x)
def get_random_example():
    return get_example(np.random.randint(len(EXAMPLES)))

def get_example_result(idx):
    x = torch.tensor(EXAMPLE_RESULTS[idx]).cuda()
    if len(x) < TOTAL_SAMPLES_OUT:
        x = torch.cat((x, torch.tensor([0]*(TOTAL_SAMPLES_OUT-len(x))).cuda()))

    x = x[:TOTAL_SAMPLES_OUT]
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

    def generate_one_step(self):
        with torch.no_grad():
            predicted_logits = self.generator(generate_input_noise(TOTAL_SAMPLES_IN)).detach().cpu()
        return predicted_logits

def create_inputs():
        x = get_random_example()
        y = get_random_example_result()

        if INPUT_MODE == 'conv':
            y = torch.from_numpy(spectral_convolution(x, y)).cuda()
        return x, y

def generate_input_noise(b_size):
    return torch.randn(b_size, TOTAL_SAMPLES_IN, 1, 1).cuda()

gen_loss, dis_loss = None, None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def run_models(window, real):
    global gen_loss, dis_loss

    def verdict_str(out):
        if len(out) > 1:
            out = float(torch.mean(out))
        if REAL_LABEL > FAKE_LABEL:
            if REAL_LABEL-out > FAKE_LABEL+out:
                return "FAKE"
            elif REAL_LABEL-out < FAKE_LABEL+out:
                return "REAL"
            else:
                return "UNSURE"
        else:
            if REAL_LABEL-out < FAKE_LABEL+out:
                return "FAKE"
            elif REAL_LABEL-out > FAKE_LABEL+out:
                return "REAL"
            else:
                return "UNSURE"

    dis.zero_grad()
    output = dis(real).view(-1)
    window.addstr(6,0, "| Real Verdict: "+verdict_str(output)+"\t"
        + str(round(float(torch.mean(output)), 4))
    )
    total_real_verdicts.append(float(torch.mean(output)))
    b_size = output.size(0)
    label = torch.full((b_size,), REAL_LABEL, dtype=torch.float).cuda()
    errD_real = dis.criterion(label, output)
    errD_real.backward()
    D_x = output.mean().item()

    noise = generate_input_noise(TOTAL_SAMPLES_IN)
    fake = gen(noise)
    output = dis(fake).view(-1)
    label.fill_(FAKE_LABEL)
    errD_fake = dis.criterion(label, output)
    errD_fake.backward()
    D_G_z1 = output.mean().item()
    dis_loss = errD_real + errD_fake
    dis_optim.step()

    gen.zero_grad()
    output = dis(fake).view(-1)
    window.addstr(7,0, "| Fake Verdict: "+verdict_str(output)+"\t"
        + str(round(float(torch.mean(output)), 4))
    )
    total_fake_verdicts.append(float(torch.mean(output)))
    label.fill_(REAL_LABEL)
    errG = gen.criterion(label, output)
    errG.backward()
    gen_loss = errG
    D_G_z2 = output.mean().item()
    gen_optim.step()

def train_on_random(window, i, dirname):
    _, z = create_inputs()
    begin_time = time.time()
    run_models(window, z)
    total_gen_losses.append(gen_loss)
    total_dis_losses.append(dis_loss)
    window.addstr(8,0, "|] Gen Loss:\t\t"+ str(round(float(gen_loss), 4)))
    window.addstr(9,0, "|] Dis Loss:\t\t"+ str(round(float(dis_loss), 4)))

    time_diff = time.time() - begin_time
    window.addstr(10,0, "Models successfully trained in "+ str(round(time_diff, ndigits=2))+ " seconds.")
    # if SAVE_MODEL_EVERY_ITERS > 0 and i % SAVE_MODEL_EVERY_ITERS == 0:
    #     cprint("Saving checkpoints for models...")
    #     torch.save(gen, os.path.join(MODEL_DIR, "gen_ckpts"))
    #     torch.save(dis, os.path.join(MODEL_DIR, "dis_ckpts"))
    return time_diff

clear = lambda: os.system('cls' if os.name=='nt' else 'clear')

def train_until_interrupt(window, save_plots=False):
    states = None
    i = 0
    j = 0
    diffs = []
    start_time = time.time()
    last_time = start_time
    num_times_saved = 0
    time_since_last_save = 0
    time_last_saved = 0
    iters_per_sec = 1
    secs_per_iter = 0
    time_passed = 0
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join(TRAINING_DIR, "training_"+timestamp)

    clear()

    window.addstr(0,0, "="*80)
    window.addstr(1,0, "MODEL TRAINING BEGINS AT "+timestamp)
    window.addstr(2,0, "="*80)

    if MAX_ITERS == 0:
        i = -1
    while i < MAX_ITERS:
        if time_passed > RUN_FOR_SEC:
            break
        try:
            window.addstr(4,0, "*"*40)
            window.addstr(5,0, "Initiating iteration #"+str(j+1))
            secs_per_iter = train_on_random(window, j, dirname)
            time_passed = time.time() - start_time
            diffs.append(time_passed - last_time)
            last_time = time_passed
            if len(diffs) > 2:
                diffs = diffs[-2:]
            window.addstr(11,0, "Running time (seconds): "+ str(round(time_passed)))
            if len(diffs) > 0:
                iters_per_sec = round(len(diffs) / sum(diffs))
                window.addstr(12,0, "Iterations/sec:"+ str(iters_per_sec))
            avg_iters_per_sec = j / time_passed
            if SAVE_EVERY_SECONDS > 0 and avg_iters_per_sec > 0 and j / avg_iters_per_sec % float(SAVE_EVERY_SECONDS) < 1.0 / avg_iters_per_sec:
                time_since_last_save = -1
                #cprint("Generating progress update...")
                #time.sleep(0.5)
                if num_times_saved == 0:
                    os.mkdir(dirname)
                begin_time = time.time()
                out1 = onestep.generate_one_step()
                amp, phase = my_hilbert(out1.numpy())
                if len(total_amps) <= num_times_saved:
                    total_amps.append(amp)
                if len(total_phases) <= num_times_saved:
                    total_phases.append(phase)
                if save_plots:
                    plot_metrics(j, show=False)
                write_normalized_audio_to_disk(out1, dirname+'/progress'+str(j)+"_"+str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))+'.wav')
                time_last_saved = time.time()
                time_diff = time.time() - begin_time
                num_times_saved += 1
                #cprint("Progress update generated in", str(round(time_diff, ndigits=2)), "seconds.")
            time_since_last_save = time.time() - time_last_saved
            if MAX_ITERS != 0:
                i += 1
            window.refresh()
            if MAX_ITERS_PER_SEC > 0:
                time.sleep((1/(MAX_ITERS_PER_SEC)) - time_passed % (1/(MAX_ITERS_PER_SEC)))
            if SLEEP_TIME > 0:
                time.sleep(SLEEP_TIME)
            j += 1
        except KeyboardInterrupt:
            break
    # cprint("Saving weights to disk...")
    # timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    # dirname = os.path.join(MODEL_DIR, "gen_weights", timestamp)
    # os.mkdir(dirname)
    # gen.save_weights(os.path.join(dirname, "gen_weights"))
    # dirname = os.path.join(MODEL_DIR, "dis_weights", timestamp)
    # os.mkdir(dirname)
    # dis.save_weights(os.path.join(dirname, "dis_weights"))
    window.addstr(14,0, "="*80)
    window.addstr(15,0, "MODEL TRAINING FINISHED AT "+str(timestamp))
    window.addstr(16,0, "="*80)
    window.refresh()
    time.sleep(1)
    return i

v_cprint("Opening examples...")
EXAMPLE_FILES = open_truncate_pad(EXAMPLES_DIR)
v_cprint("Opening example results...")
EXAMPLE_RESULT_FILES = open_truncate_pad(EXAMPLE_RESULTS_DIR)
v_cprint("Opening inputs...")
INPUT_FILES = open_truncate_pad(INPUTS_DIR)
v_cprint("We have", len(INPUT_FILES), "inputs in the folder.")

print("Resampling, stand by...")
EXAMPLES = iterate_and_resample(EXAMPLE_FILES)
INPUTS = iterate_and_resample(INPUT_FILES)
EXAMPLE_RESULTS = iterate_and_resample(EXAMPLE_RESULT_FILES)
v_cprint("Created", len(EXAMPLES), "Example Arrays and", len(EXAMPLE_RESULTS), "Example Result Arrays.")

gen = TA_Generator().cuda()
dis = DPAM_Discriminator().cuda()
gen.apply(weights_init)
dis.apply(weights_init)
gen_optim = torch.optim.Adam(gen.parameters(), lr=GENERATOR_LR, betas=(GENERATOR_BETA, 0.999))
dis_optim = torch.optim.Adam(dis.parameters(), lr=DISCRIMINATOR_LR, betas=(DISCRIMINATOR_BETA, 0.999))

onestep = OneStep(gen)

if __name__ == "__main__":
    i = curses.wrapper(train_until_interrupt, True)
    print("Generating...")
    if USE_REAL_AUDIO:
        data = onestep.generate_one_step()
    else:
        data = onestep.generate_one_step()
    print("Done!")
    amp, phase = my_hilbert(data)
    total_amps.append(amp)
    total_phases.append(phase)

    write_normalized_audio_to_disk(data, 'out1.wav')

    plot_metrics(i)
