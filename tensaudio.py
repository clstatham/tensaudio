import collections
import os
import time
from datetime import datetime
import curses

import ctcsound
import librosa
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
import matplotlib.backends.backend_agg as agg
import numpy as np
import scipy.io.wavfile
import six
import soundfile
import torch
import torch.nn as nn

from pgvis import init_gvis
from csoundinterface import CsoundInterface, CSIRun, G_csi
from discriminator import TADiscriminator
from generator import TAGenerator, TAInstParamGenerator
from global_constants import *
from helper import *
from hilbert import *

torch.autograd.set_detect_anomaly(True)

np.random.seed(int(round(time.time())))
torch.random.seed()


total_amps = []
total_phases = []
total_gen_losses = []
total_dis_losses = []
total_real_verdicts = []
total_fake_verdicts = []

# EXAMPLE_RESULT_FILES = []
# EXAMPLE_FILES = []
# INPUT_FILES = []
# EXAMPLE_RESULTS = []
# EXAMPLES = []
# INPUTS = []
# gen = None
# dis = None
# gen_optim = None
# dis_optim = None
# onestep = None
# G_vis = None

def record_amp_phase(amp, phase):
    # total_amps.append(amp)
    # total_phases.append(phase)
    pass
def plot_metrics(i, save_to_disk=False):
    global total_gen_losses, total_dis_losses
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join(PLOTS_DIR, timestamp)

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

    if type(total_gen_losses) is torch.Tensor:
        total_gen_losses = total_gen_losses.detach().cpu().numpy()
    if type(total_dis_losses) is torch.Tensor:
        total_dis_losses = total_dis_losses.detach().cpu().numpy()
    if len(total_gen_losses) + len(total_dis_losses) + len(total_real_verdicts) + len(total_fake_verdicts) > 0:
        fig3, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=[8,6], dpi=100)
        fig3.subplots_adjust(hspace=1)
        fig3.suptitle("Gen/Dis Losses " + timestamp)
        ax1.set_title("Gen Losses")
        ax1.plot(range(len(total_gen_losses)), total_gen_losses, color="b")
        ax2.set_title("Dis Losses")
        ax2.plot(range(len(total_dis_losses)), total_dis_losses, color="r")
        ax3.set_title("Real/Fake Verdicts")
        ax3.plot(range(len(total_real_verdicts)), total_real_verdicts, label="Real", color="g")
        ax3.plot(range(len(total_fake_verdicts)), total_fake_verdicts, label="Fake", color="m")
        ax3.legend()
        canvas = agg.FigureCanvasAgg(fig3)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        if save_to_disk and i % SAVE_EVERY_SECONDS == 0:
            os.mkdir(dirname)
            fig3.savefig(os.path.join(dirname, filename3))
    size = canvas.get_width_height()
    plt.close()
    return raw_data, size

def open_truncate_pad(name):
    ret = []
    srs = torch.empty(0)
    for filename in os.listdir(os.path.join(RESOURCES_DIR, name)):
        f = soundfile.SoundFile(os.path.join(RESOURCES_DIR, name, filename), "r")
        data = torch.as_tensor(f.read()).requires_grad_(True)
        sr = torch.as_tensor(f.samplerate).unsqueeze(0)
        #stacked = torch.stack((data, sr)).requires_grad_(True)
        ret.append(data)
        srs = torch.cat((srs.detach().requires_grad_(True), sr))
    return ret, srs

def iterate_and_resample(files, srs):
    arr = []
    for i in range(len(files)):
        fil = files[i]
        if len(fil.shape) > 1 and fil.shape[1] != 1:
            fil = fil.permute(1,0)[0]
        fortfil = np.asfortranarray(fil.detach().cpu().numpy())
        a = librosa.resample(fortfil, srs[i].item(), SAMPLE_RATE)
        a = torch.from_numpy(a).requires_grad_(True)
        arr.append(a)
    return arr


def select_input(idx):
    x = INPUTS[idx]
    if len(x) < TOTAL_SAMPLES_OUT:
        x = np.concatenate((x, [0]*(TOTAL_SAMPLES_OUT-len(x))))
    
    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_audio(x)

def get_example(idx):
    x = EXAMPLES[idx]
    if len(x) < TOTAL_SAMPLES_OUT:
        x = np.concatenate((x, [0]*(TOTAL_SAMPLES_OUT-len(x))))

    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_audio(x)
def get_random_example():
    return get_example(np.random.randint(len(EXAMPLES)))

def get_example_result(idx):
    x = EXAMPLE_RESULTS[idx]
    if len(x) < TOTAL_SAMPLES_OUT:
        x = np.concatenate((x, [0]*(TOTAL_SAMPLES_OUT-len(x))))

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
            predicted_logits = self.generator(generate_input_noise(TOTAL_SAMPLES_IN))
        return predicted_logits

def create_input():
        #x = get_random_example()
        y = get_random_example_result()

        #if INPUT_MODE == 'conv':
        #    y = torch.from_numpy(spectral_convolution(x, y)).cuda()
        return y

def generate_input_noise(b_size):
    if GEN_MODE in [5]:
        return torch.randn(1, TOTAL_SAMPLES_IN, 1, requires_grad=True).cuda()
    else:
        return torch.randn(b_size, TOTAL_SAMPLES_IN, 1, 1, requires_grad=True).cuda()

gen_loss, dis_loss = None, None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_output_from_params(params, window):
    global current_params
    #params = params.clone().cpu().numpy()
    current_params = params
    #if window is not None:
        #window.addstr(3,0, str(params.shape[0]))
        #window.refresh()
    audio, _ = CSIRun.apply(params.clone())
    #noise = np.random.randn(len(audio)) * 0.0001
    # for i in range(len(audio)):
    #     if audio[i] == np.inf or audio[i] == np.nan:
    #         audio[i] = 0.00001
    return audio

def run_models(window, real):
    #global gen_loss, dis_loss

    def verdict_str(out):
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

    #real_label = torch.as_tensor(REAL_LABEL).to(torch.float).cuda()
    #fake_label = torch.as_tensor(FAKE_LABEL).to(torch.float).cuda()

    dis.zero_grad()

    y = dis(real)
    
    real_verdict = y.mean()
    label = torch.full(y.shape, REAL_LABEL).to(torch.float).cpu()
    dis_loss_real = dis.criterion(label, y)
    #dis_loss_real.backward(retain_graph=True)

    if GEN_MODE in [5]:
        noise = generate_input_noise(1)
        params = gen(noise)
        audio = get_output_from_params(params, window)
        fake = audio.float().cuda()
    else:
        noise = generate_input_noise(TOTAL_SAMPLES_IN)
        fake = gen(noise, window)
    y = dis(fake)
    
    fake_verdict = y.mean()
    label = torch.full(y.shape, FAKE_LABEL).to(torch.float).cpu()
    dis_loss_fake = dis.criterion(label, y)
    #dis_loss_fake.backward(retain_graph=True)
    dis_loss = dis_loss_real + dis_loss_fake
    dis_loss.backward(retain_graph=True)
    dis_optim.step()
    
    gen.zero_grad()

    y = dis(fake)
    fake_verdict = y.mean()
    label = torch.full(y.shape, REAL_LABEL).to(torch.float).cuda()
    gen_loss = gen.criterion(label, y)
    #gradients = torch.autograd.grad(outputs=fake_verdict, inputs=gen_loss, grad_outputs=real_label, create_graph=False, retain_graph=None, only_inputs=True)[0]
    #gp = ((gradients.norm(dim=1)-1)**2).mean() + fake_verdict
    gen_loss.backward()
    #gp.backward()
    gen_optim.step()

    #dis_params = str([p.grad for p in list(dis.parameters())])
    #gen_params = str([p.grad for p in list(gen.parameters())])
    #print(gen_params)
    #print("+++++++++++++++++++++++++++++++++++++++++")
    #print(dis_params)
    

    try:
        window.move(6,0)
        window.clrtoeol()
        window.move(7,0)
        window.clrtoeol()
        window.addstr(6,0, "| Real Verdict: "+verdict_str(real_verdict.item())+"\t"
            + str(round(real_verdict.item(), 4)))
        window.addstr(7,0, "| Fake Verdict: "+verdict_str(fake_verdict.item())+"\t"
            + str(round(fake_verdict.item(), 4)))
        window.refresh()
    except curses.error:
        pass


    total_real_verdicts.append(real_verdict.item())
    total_fake_verdicts.append(fake_verdict.item())

    return gen_loss, dis_loss

def train_on_random(window, i, dirname):
    z = torch.autograd.Variable(torch.as_tensor(create_input()), requires_grad=True).cuda()
    begin_time = time.time()
    gen_loss, dis_loss = run_models(window, z)
    total_gen_losses.append(gen_loss.clone().detach().cpu().numpy())
    total_dis_losses.append(dis_loss.clone().detach().cpu().numpy())
    window.move(8,0)
    window.clrtoeol()
    window.move(9,0)
    window.clrtoeol()
    window.addstr(8,0, "|] Gen Loss:\t\t"+ str(round(float(gen_loss), 4)))
    window.addstr(9,0, "|] Dis Loss:\t\t"+ str(round(float(dis_loss), 4)))

    time_diff = time.time() - begin_time
    window.move(10,0)
    window.clrtoeol()
    window.addstr(10,0, "Time per iteration: "+ str(round(time_diff, ndigits=2))+ " seconds.")
    window.refresh()
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

    #clear()

    window.addstr(0,0, "="*80)
    window.addstr(1,0, "MODEL TRAINING STARTED AT "+timestamp)
    window.addstr(2,0, "="*80)

    #csi.start()

    if MAX_ITERS == 0:
        i = -1
    while i < MAX_ITERS:
        if RUN_FOR_SEC > 0 and time_passed > RUN_FOR_SEC:
            break
        try:
            G_vis.handle_events()
            window.addstr(4,0, "*"*40)
            window.addstr(5,0, "Initiating iteration #"+str(j+1))
            
            #dis_optim.zero_grad()
            #gen_optim.zero_grad()

            secs_per_iter = train_on_random(window, j, dirname)
            time_passed = time.time() - start_time
            diffs.append(time_passed - last_time)
            last_time = time_passed
            if len(diffs) > 2:
                diffs = diffs[-2:]
            try:
                window.addstr(11,0, "Running time (seconds): "+ str(round(time_passed)))
                window.refresh()
            except curses.error:
                pass
            if len(diffs) > 0:
                iters_per_sec = round(len(diffs) / sum(diffs), 3)
                try:
                    window.move(12,0)
                    window.clrtoeol()
                    window.addstr(12,0, "Iterations/sec:"+ str(iters_per_sec))
                    window.refresh()
                except curses.error:
                    pass
            avg_iters_per_sec = j / time_passed
            if save_plots:
                img, size = plot_metrics(j, save_to_disk=False)
                G_vis.draw_metrics(img, size)
            if SAVE_EVERY_SECONDS > 0 and avg_iters_per_sec > 0 and j / avg_iters_per_sec % float(SAVE_EVERY_SECONDS) < 1.0 / avg_iters_per_sec:
                time_since_last_save = -1
                #cprint("Generating progress update...")
                #time.sleep(0.5)
                if num_times_saved == 0:
                    os.mkdir(dirname)
                begin_time = time.time()
                if save_plots:
                    img, size = plot_metrics(j, save_to_disk=True)
                if GEN_MODE in [5]:
                    params = onestep.generate_one_step()
                    out1 = get_output_from_params(params, window)
                    #amp, phase = my_hilbert(out1)
                else:
                    out1 = onestep.generate_one_step()
                    #amp, phase = my_hilbert(out1)
                #if len(total_amps) <= num_times_saved:
                #    total_amps.append(amp)
                #if len(total_phases) <= num_times_saved:
                #    total_phases.append(phase)
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
    try:
        window.addstr(14,0, "="*80)
        window.addstr(15,0, "MODEL TRAINING FINISHED AT "+str(timestamp))
        window.addstr(16,0, "="*80)
        window.refresh()
    except curses.error:
        pass
    #time.sleep(1)
    return i


if __name__ == "__main__":
    global EXAMPLE_RESULT_FILES, EXAMPLE_FILES, INPUT_FILES
    global EXAMPLE_RESULTS, EXAMPLES, INPUTS
    global gen, dis, gen_optim, dis_optim, onestep
    print("Opening example results...")
    EXAMPLE_RESULT_FILES, EXAMPLE_RESULT_SRS = open_truncate_pad(EXAMPLE_RESULTS_DIR)
    if USE_REAL_AUDIO:
        print("Opening examples...")
        EXAMPLE_FILES = open_truncate_pad(EXAMPLES_DIR)
        print("Opening inputs...")
        INPUT_FILES = open_truncate_pad(INPUTS_DIR)
        print("We have", len(INPUT_FILES), "inputs in the folder.")

    print("Resampling, stand by...")
    EXAMPLE_RESULTS = iterate_and_resample(EXAMPLE_RESULT_FILES, EXAMPLE_RESULT_SRS)
    if USE_REAL_AUDIO:
        EXAMPLES = iterate_and_resample(EXAMPLE_FILES)
        INPUTS = iterate_and_resample(INPUT_FILES)
    if USE_REAL_AUDIO:
        print("Created", len(EXAMPLES), "Example Arrays and", len(EXAMPLE_RESULTS), "Example Result Arrays.")
    else:
        print("Created", len(EXAMPLE_RESULTS), "Example Result Arrays.")

    print("Creating networks...")
    gen = TAInstParamGenerator().cuda()
    dis = TADiscriminator().cpu()
    gen.apply(weights_init)
    dis.apply(weights_init)
    gen_optim = torch.optim.Adam(gen.parameters(), lr=GENERATOR_LR, betas=(GENERATOR_BETA, 0.999))
    dis_optim = torch.optim.Adam(dis.parameters(), lr=DISCRIMINATOR_LR, betas=(DISCRIMINATOR_BETA, 0.999))

    onestep = OneStep(gen)

    print_global_constants()
    
    if GEN_MODE in [5]:
        print("Initializing Visualizer...")
        G_vis = init_gvis(800, 600)

        print("Creating CSound Interface...")
        result = G_csi.compile()
        if result != 0:
            raise RuntimeError("CSound compilation failed!")
        #G_vis.watch_val(csi.param_gen(1))
        #for i in range(10):
        #    G_vis.watch_val(csi.param_gen(40+i))
        current_params = [0.]*N_PARAMS*TOTAL_PARAM_UPDATES

    print("Initialization complete! Starting...")
    time.sleep(1)

    i = curses.wrapper(train_until_interrupt, True)
    print("Generating...")
    if GEN_MODE in [5]:
        params = onestep.generate_one_step()
        data = get_output_from_params(params, None)
    else:
        data = onestep.generate_one_step()
    print("Done!")
    G_csi.stop()
    amp, phase = my_hilbert(data)
    total_amps.append(amp)
    total_phases.append(phase)

    write_normalized_audio_to_disk(data, 'out1.wav')

    plot_metrics(i, save_to_disk=True)
