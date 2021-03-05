import collections
import curses
import os
import time
from datetime import datetime

import ctcsound
import librosa
import librosa.display
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pygame
import scipy.io.wavfile
import six
import soundfile
import timer3
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import animation, cm
from pygame.locals import *

from csoundinterface import CSIRun, CsoundInterface, G_csi
from discriminator import TADiscriminator
from generator import TAGenerator, TAInstParamGenerator
from global_constants import *
from helper import *
from hilbert import *

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#torch.autograd.set_detect_anomaly(True)
plt.switch_backend('agg')

np.random.seed(int(round(time.time())))
torch.random.manual_seed(int(round(time.time())))

total_amps = []
total_phases = []
total_gen_losses = []
total_dis_losses = []
total_real_verdicts = []
total_fake_verdicts = []
current_view_mode = 1
current_output = np.zeros(TOTAL_SAMPLES_OUT)
current_example = np.zeros(TOTAL_SAMPLES_OUT)
vis_paused = True

def clear_metrics():
    global total_gen_losses, total_dis_losses, total_fake_verdicts, total_real_verdicts
    total_dis_losses = []
    total_fake_verdicts = []
    total_gen_losses = []
    total_real_verdicts = []

def record_amp_phase(amp, phase):
    # total_amps.append(amp)
    # total_phases.append(phase)
    pass

@torch.no_grad()
def plot_metrics(i, save_to_disk=False):
    global total_gen_losses, total_dis_losses
    timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
    dirname = os.path.join(PLOTS_DIR, timestamp)

    filename1 = str('AMP_'+timestamp+'_'+str(i)+'.png')
    filename2 = str('PHASE_'+timestamp+'_'+str(i)+'.png')
    filename3 = str('LOSSES_'+timestamp+'_'+str(i)+'.png')
    filename4 = str('SPECTROGRAMS_'+timestamp+'_'+str(i)+'.png')

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
        total_gen_losses = total_gen_losses.clone().detach().cpu().numpy()
    if type(total_dis_losses) is torch.Tensor:
        total_dis_losses = total_dis_losses.clone().detach().cpu().numpy()
    if len(total_gen_losses) + len(total_dis_losses) + len(total_real_verdicts) + len(total_fake_verdicts) > 0:
        fig3, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=[VIS_WIDTH//100,VIS_HEIGHT//50], dpi=50, facecolor="white")
        fig3.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, bottom=0.05, right=0.95, top=0.95)
        plt.title("Gen/Dis Losses " + timestamp)
        ax1.set_title("Gen Losses")
        ax1.set_facecolor("white")
        ax1.plot(range(len(total_gen_losses)), total_gen_losses, color="b")
        ax2.set_title("Dis Losses")
        ax2.set_facecolor("white")
        ax2.plot(range(len(total_dis_losses)), total_dis_losses, color="r")
        ax3.set_title("Real/Fake Verdicts")
        ax3.set_facecolor("white")
        ax3.plot(range(len(total_real_verdicts)), total_real_verdicts, label="Real", color="g")
        ax3.plot(range(len(total_fake_verdicts)), total_fake_verdicts, label="Fake", color="m")
        ax3.legend()
        canvas = agg.FigureCanvasAgg(fig3)
        canvas.draw()
        renderer = canvas.get_renderer()
        plots_img = renderer.tostring_rgb()
        
        size_plots = canvas.get_width_height()
        plt.close()

        surf_plots = pygame.image.fromstring(plots_img, size_plots, "RGB")
        ta_surface.blit(surf_plots, (0,0))
        #spec_out = librosa.feature.melspectrogram(current_output, SAMPLE_RATE, n_fft=N_FFT, hop_length=64)
        #librosa.display.specshow(librosa.power_to_db(spec_out, ref=np.max), y_axis='mel', fmax=SAMPLE_RATE/2, x_axis='time')
        #out_spec_fig = plt.figure(figsize=[VIS_WIDTH//100, VIS_HEIGHT//50], dpi=50)
        if current_view_mode == 3:
            out_spec_fig, spec_ax3 = plt.subplots(1, 1, figsize=[VIS_WIDTH//50, VIS_HEIGHT//50], dpi=50, facecolor="white")
        else:
            out_spec_fig, ((spec_ax1, spec_ax2), (spec_ax3, spec_ax4)) = plt.subplots(2, 2, figsize=[VIS_WIDTH//100, VIS_HEIGHT//50], dpi=50, facecolor="white")
        out_spec_fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, bottom=0.05, right=0.95, top=0.95)

        if current_view_mode == 1:
            spec_ax1.set_title('Output Rainbowgram')
            spec_ax3.set_title('Output Mel Spectrogram')
        elif current_view_mode == 2:
            spec_ax1.set_title('Progress Rainbowgram')
            spec_ax3.set_title('Progress Mel Spectrogram')
        else:
            spec_ax3.set_title('Progress Mel Spectrogram')
        if GEN_MODE == 4 and DIS_MODE == 2:
            global current_output
            current_output = MelToSTFTWithGradients.apply(torch.from_numpy(current_output), N_GEN_MEL_CHANNELS)
            current_output = stft_to_audio(current_output, GEN_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_SAVING).detach().clone().cpu().numpy()
        if DIS_MODE == 2:
            global current_example
            current_example = MelToSTFTWithGradients.apply(torch.from_numpy(current_example), DIS_N_MELS)
            current_example = stft_to_audio(current_example, DIS_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_SAVING).detach().clone().cpu().numpy()
        if current_view_mode != 3:
            spec_ax2.set_title('Example Rainbowgram')
            spec_ax4.set_title('Example Mel Spectrogram')
            note_specgram(current_output, spec_ax1, VIS_HOP_LEN)
            note_specgram(current_example, spec_ax2, VIS_HOP_LEN)
            spec_ax4.specgram(current_example, VIS_N_FFT, noverlap=VIS_N_FFT//2, mode='psd', cmap='magma')
        
        spec_ax3.specgram(current_output, VIS_N_FFT, noverlap=VIS_N_FFT//2, mode='psd', cmap='magma')
        
        canvas = agg.FigureCanvasAgg(out_spec_fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        out_spec_img = renderer.tostring_rgb()
        size_out_spec_img = canvas.get_width_height()
        plt.close()
        out_spec_surf = pygame.image.fromstring(out_spec_img, size_out_spec_img, "RGB")
        if current_view_mode == 3:
            ta_surface.blit(out_spec_surf, (0, 0))
        else:
            ta_surface.blit(out_spec_surf, (VIS_WIDTH//2, 0))
        
        pygame.display.flip()

        if save_to_disk:
            os.mkdir(dirname)
            fig3.savefig(os.path.join(dirname, filename3))
            out_spec_fig.savefig(os.path.join(dirname, filename4))
        #yield

def open_truncate_pad(name):
    ret = []
    srs = torch.empty(0)
    for filename in os.listdir(os.path.join(RESOURCES_DIR, name)):
        f = soundfile.SoundFile(os.path.join(RESOURCES_DIR, name, filename), "r")
        data = torch.as_tensor(f.read()).requires_grad_(True)
        sr = torch.as_tensor(f.samplerate).unsqueeze(0)
        #stacked = torch.stack((data, sr)).requires_grad_(True)
        if len(data.shape) > 1 and data.shape[1] != 1:
            data = data.permute(1,0)[0]
        sr_quotient = f.samplerate / SAMPLE_RATE
        new_len = OUTPUT_DURATION * sr_quotient
        new_samples = int(new_len * f.samplerate)
        ret.append(data[:new_samples])
        srs = torch.cat((srs.detach().requires_grad_(True), sr))
    return ret, srs

def iterate_and_resample(files, srs):
    arr = []
    for i in range(len(files)):
        fil = files[i]
        if len(fil.shape) > 1 and fil.shape[1] != 1:
            fil = fil.permute(1,0)[0]
        #fortfil = np.asfortranarray(fil.detach().requires_grad_(True).cpu().numpy())
        #a = librosa.resample(fortfil, srs[i].item(), SAMPLE_RATE)
        a = torchaudio.transforms.Resample(srs[i].item(), SAMPLE_RATE).forward(fil.detach().clone())
        # = torch.from_numpy(a).requires_grad_(True)
        arr.append(a[:TOTAL_SAMPLES_OUT].float())
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
        x = torch.cat((x, torch.zeros(TOTAL_SAMPLES_OUT-len(x))))

    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_audio(x)
def get_random_example_result():
    return get_example_result(np.random.randint(len(EXAMPLE_RESULTS)))



class OneStep():
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    @torch.no_grad()
    def generate_one_step(self, noise=None):
        if noise is None:
            noise = generate_input_noise().to(0)
        self.generator.eval()
        predicted_logits = self.generator(noise.detach().clone())
        self.generator.train()
        return torch.squeeze(predicted_logits)

def create_input():
        #x = get_random_example()
        y = normalize_audio(torch.as_tensor(get_random_example_result())).to(0)
        if DIS_MODE == 2:
            y = AudioToMelWithGradients.apply(y, DIS_N_FFT, DIS_N_MELS, DIS_HOP_LEN)
        #if INPUT_MODE == 'conv':
        #    y = torch.from_numpy(spectral_convolution(x, y)).to(0)
        return y

def generate_input_noise():
    if GEN_MODE in [10]:
        return torch.randn(1, TOTAL_SAMPLES_IN, 1, requires_grad=True).to(0)
    elif GEN_MODE in [2]:
        return torch.randn(BATCH_SIZE, 2, TOTAL_SAMPLES_IN, requires_grad=True).to(0)
    elif GEN_MODE in [4]:
        return torch.randn(BATCH_SIZE, 1, N_GEN_MEL_CHANNELS, TOTAL_SAMPLES_IN, requires_grad=True).to(0)
    elif GEN_MODE in [6]:
        return torch.randn(BATCH_SIZE, 2, N_GEN_FFT // 4, TOTAL_SAMPLES_IN, requires_grad=True).to(0)
    else:
        return torch.randn(BATCH_SIZE, N_CHANNELS, TOTAL_SAMPLES_IN, requires_grad=True).to(0)
    
training_noise = ag.Variable(generate_input_noise(), requires_grad=True)
gen_loss, dis_loss = None, None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Norm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def get_output_from_params(params, window):
    global current_params
    #params = params.clone().cpu().numpy()
    current_params = params
    #if window is not None:
        #window.addstr(3,0, str(params.shape[0]))
        #window.refresh()
    audio = CSIRun.apply(params.clone())
    #noise = np.random.randn(len(audio)) * 0.0001
    # for i in range(len(audio)):
    #     if audio[i] == np.inf or audio[i] == np.nan:
    #         audio[i] = 0.00001
    return audio

def run_models(window, real):
    #global gen_loss, dis_loss

    def verdict_str(out1, out2):
        if REAL_LABEL > FAKE_LABEL:
            if out1 > out2:
                return "REAL", "FAKE"
            elif out1 < out2:
                return "FAKE", "REAL"
            else:
                return "UNSURE", "UNSURE"
        else:
            if out1 < out2:
                return "REAL", "FAKE"
            elif out1 > out2:
                return "FAKE", "REAL"
            else:
                return "UNSURE", "UNSURE"

    dis.zero_grad()
    
    verdict = dis(real).view(-1)
    label = torch.full(verdict.shape, REAL_LABEL).to(torch.float).to(0)
    dis_loss_real = dis.criterion(label, verdict)
    #dis_loss_real.backward()
    real_verdict = verdict.flatten().squeeze().mean().item()

    if GEN_MODE in [5]:
        noise = generate_input_noise()
        params = gen(noise)
        audio = get_output_from_params(params, window)
        fake = torch.squeeze(audio).float().to(0)
    else:
        noise = generate_input_noise()
        fake = gen(noise)
    
    verdict = dis(fake.detach()).view(-1)
    fake_verdict = verdict.flatten().squeeze().mean()
    if torch.isnan(fake_verdict):
        raise RuntimeError("Discriminator output NaN!")
    label = torch.full(verdict.shape, FAKE_LABEL).to(torch.float).to(0)
    dis_loss_fake = dis.criterion(label, verdict)
    #dis_loss_fake.backward()
    dis_loss = dis_loss_real + dis_loss_fake
    dis_loss.backward()
    dis_optim.step()

    gen.zero_grad()
    verdict = dis(fake).view(-1)
    
    label = torch.full(verdict.shape, REAL_LABEL).to(torch.float).to(0)
    #label = torch.full(y3.shape, REAL_LABEL).to(torch.float).to(0)
    gen_loss = gen.criterion(label, verdict)
    gen_loss.backward()
    fake_verdict = verdict.flatten().squeeze().mean().item()
    gen_optim.step()

    real_str, fake_str = verdict_str(real_verdict, fake_verdict)
    try:
        window.addstr(6,0, "| Real Verdict: "+real_str+"\t"
            + str(round(real_verdict, 4)))
        window.addstr(7,0, "| Fake Verdict: "+fake_str+"\t"
            + str(round(fake_verdict, 4)))
        #window.addstr(8,0, "| Verdict Diff: \t"+"\t"
        #    + str(round(gen_loss.flatten().squeeze().mean().item(), 4)))
        window.refresh()
    except curses.error:
        pass


    total_real_verdicts.append(real_verdict)
    total_fake_verdicts.append(fake_verdict)

    return real, fake, gen_loss.flatten().squeeze().mean(), dis_loss

def save_states(epoch):
    torch.save({
        'epoch': epoch,
        'gen_state': gen.state_dict(),
        'dis_state': dis.state_dict(),
        'gen_optim_state': gen_optim.state_dict(),
        'dis_optim_state': dis_optim.state_dict(),
    }, os.path.join(MODEL_DIR, "checkpoints", "checkpoint.pt"))

def train_on_random(window, epoch, dirname):
    global current_output, current_example
    while True:
        z = torch.autograd.Variable(create_input(), requires_grad=True).to(0)
        begin_time = time.time()
        real, fake, gen_loss, dis_loss = run_models(window, z)
        if current_view_mode == 1:
            current_output = fake.detach().clone().cpu().numpy()
        elif current_view_mode in [2,3]:
            current_output = onestep.generate_one_step(training_noise).detach().clone().cpu().numpy()
        current_example = real.detach().clone().cpu().numpy()
        total_gen_losses.append(gen_loss.item())
        total_dis_losses.append(dis_loss.item())

        try:
            window.addstr(8,0, "|] Gen Loss:\t\t"+ str(round(float(gen_loss), 4)))
            window.addstr(9,0, "|] Dis Loss:\t\t"+ str(round(float(dis_loss), 4)))
        except curses.error:
            pass

        time_diff = time.time() - begin_time
        #window.move(10,0)
        #window.clrtoeol()
        window.addstr(10,0, "Time per iteration: "+ str(round(time_diff, ndigits=2))+ " seconds.")
        window.refresh()
        if SAVE_MODEL_EVERY_ITERS > 0 and epoch % SAVE_MODEL_EVERY_ITERS == 0:
            #window.move(10,40)
            #window.clrtoeol()
            window.addstr(10,40, "Saving model states...")
            window.refresh()
            save_states(epoch)
        else:
            pass
            #window.move(10,40)
            #window.clrtoeol()
            #window.refresh()
        yield time_diff

clear = lambda: os.system('cls' if os.name=='nt' else 'clear')

def generate_progress_report(window, epoch, dirname, force=False):
    #cprint("Generating progress update...")
    #time.sleep(0.5)
    try:
        window.move(11,40)
        window.clrtoeol()
        window.addstr(11,40, "Generating progress update...")
        window.refresh()
    except curses.error:
        pass
    try:
        os.mkdir(dirname)
    except:
        pass
    begin_time = time.time()
    if (not vis_paused) or force:
        plot_metrics(epoch, save_to_disk=True)
    if GEN_MODE in [5]:
        params = onestep.generate_one_step(training_noise)
        out1 = get_output_from_params(params, window)
    elif GEN_MODE in [4] and DIS_MODE in [2]:
        melspec = onestep.generate_one_step(training_noise)
        stft = MelToSTFTWithGradients.apply(melspec[:,:], N_GEN_MEL_CHANNELS)
        out1 = stft_to_audio(stft, VIS_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_SAVING)
    else:
        out1 = onestep.generate_one_step(training_noise)
    
    write_normalized_audio_to_disk(out1, dirname+'/progress'+"_"+str(datetime.now().strftime("%H.%M.%S"))+'_'+str(epoch)+'.wav')
    time_last_saved = time.time()
    time_diff = time.time() - begin_time
    print("Progress update generated in", str(round(time_diff, ndigits=2)), "seconds.")
    try:
        window.move(11,40)
        window.clrtoeol()
        window.refresh()
    except curses.error:
        pass
    return time_last_saved

def train_until_interrupt(window, starting_epoch, save_plots=False):
    states = None
    i = 0
    epoch = starting_epoch
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
    dirname = TRAINING_DIR+'/'+datetime.now().strftime("%d.%m.%Y")

    #clear()
    for x in range(15):
        try:
            window.move(x,0)
            window.clrtoeol()
        except curses.error:
            pass
    window.refresh()
    
    if MAX_ITERS == 0:
        i = -1
    while i < MAX_ITERS:
        if RUN_FOR_SEC > 0 and time_passed > RUN_FOR_SEC:
            break
        try:
            try:
                window.addstr(0,0, "="*80)
                window.addstr(1,0, "MODEL TRAINING STARTED AT "+timestamp)
                window.addstr(2,0, "="*80)
                window.addstr(4,41, "Visualizer key commands:")
                window.addstr(5,41, "s = save checkpoint")
                window.addstr(6,41, "c = clear metrics")
                window.addstr(7,41, "x = save & exit")
                window.addstr(8,41, "r = generate progress report")
                window.addstr(9,41, "p = pause visualization (models will still run)")
                window.addstr(10,41, "m = change view mode (output/progress)")
                window.refresh()
            except curses.error:
                pass
            try:
                pygame.event.pump()
                for ev in pygame.event.get():
                    if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_x):
                        window.move(10,40)
                        window.clrtoeol()
                        window.addstr(10,40, "Saving model states...")
                        window.refresh()
                        save_states(epoch)
                        return i
                    elif ev.type == KEYDOWN:
                        if ev.key == K_s:
                            window.move(10,40)
                            window.clrtoeol()
                            window.addstr(10,40, "Saving model states...")
                            window.refresh()
                            save_states(epoch)
                            window.move(10,40)
                            window.clrtoeol()
                            window.refresh()
                        elif ev.key == K_c:
                            clear_metrics()
                        elif ev.key == K_r:
                            generate_progress_report(window, epoch, dirname, force=True)
                            #write_normalized_audio_to_disk(out1, dirname1+'/progress'+str(datetime.now().strftime("%H.%M.%S"))+'_'+str(epoch)+'.wav')
                            num_times_saved += 1
                            time_since_last_save = -1
                        elif ev.key == K_p:
                            global vis_paused
                            vis_paused = not vis_paused
                        elif ev.key == K_m:
                            global current_view_mode
                            if current_view_mode == 1:
                                current_view_mode = 2
                            elif current_view_mode == 2:
                                current_view_mode = 3
                            elif current_view_mode == 3:
                                current_view_mode = 1
                        elif ev.key == K_SPACE:
                            plot_metrics(epoch, False)
                ta_clk.tick(60)
                pygame.display.flip()
            except:
                pass

            window.addstr(4,0, "*"*40)
            window.addstr(5,0, "Initiating iteration #"+str(epoch))
            trainfunc = train_on_random(window, epoch, dirname)
            secs_per_iter = next(trainfunc)
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
            avg_iters_per_sec = epoch / time_passed
            if save_plots and epoch % VIS_UPDATE_INTERVAL == 0 and not vis_paused:
                plot_metrics(epoch, save_to_disk=False)
            if SAVE_EVERY_SECONDS > 0 and avg_iters_per_sec > 0 and epoch / avg_iters_per_sec % float(SAVE_EVERY_SECONDS) < 1.0 / avg_iters_per_sec:
                generate_progress_report(window, epoch, dirname, force=True)
                time_since_last_save = -1
                num_times_saved += 1
            time_since_last_save = time.time() - time_last_saved
            if MAX_ITERS != 0:
                i += 1
            window.refresh()
            if MAX_ITERS_PER_SEC > 0:
                time.sleep((1/(MAX_ITERS_PER_SEC)) - time_passed % (1/(MAX_ITERS_PER_SEC)))
            if SLEEP_TIME > 0:
                time.sleep(SLEEP_TIME)
            epoch += 1
        except KeyboardInterrupt:
            break
    
    try:
        window.addstr(14,0, "="*80)
        window.addstr(15,0, "MODEL TRAINING FINISHED AT "+str(timestamp))
        window.addstr(16,0, "="*80)
        window.move(10,40)
        window.clrtoeol()
        window.addstr(10,40, "Saving model states...")
        window.refresh()
    except curses.error:
        pass
    save_states(epoch)
    return i


if __name__ == "__main__":
    global EXAMPLE_RESULT_FILES, EXAMPLE_FILES, INPUT_FILES
    global EXAMPLE_RESULTS, EXAMPLES, INPUTS
    global gen, dis, gen_optim, dis_optim, onestep, ta_surface, ta_clk
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

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dist.init_process_group('gloo', init_method="file:///D:/tensaudio/filestore", world_size=1, rank=0)

    print("Creating Generator...")
    gen = TAGenerator().to(0)
    gen_ddp = DDP(gen)
    print("Creating Discriminator...")
    dis = TADiscriminator().to(0)
    dis_ddp = DDP(dis)
    print("Creating Optimizers...")
    #gen_optim = torch.optim.SGD(gen.parameters(), lr=GENERATOR_LR, momentum=GENERATOR_MOMENTUM)
    gen_optim = torch.optim.Adam(gen.parameters(), lr=GENERATOR_LR, betas=(GENERATOR_BETA, 0.999))
    dis_optim = torch.optim.Adam(dis.parameters(), lr=DISCRIMINATOR_LR, betas=(DISCRIMINATOR_BETA, 0.999))
    #dis_optim = torch.optim.RMSprop(dis.parameters(), lr=DISCRIMINATOR_LR, momentum=DISCRIMINATOR_MOMENTUM)
    #gen = TAInstParamGenerator().to(0)
    print("Attempting to load states from disk...")
    try:
        checkpoint = torch.load(os.path.join(MODEL_DIR, "checkpoints", "checkpoint.pt"))
        gen.load_state_dict(checkpoint['gen_state'])
        dis.load_state_dict(checkpoint['dis_state'])
        gen_optim.load_state_dict(checkpoint['gen_optim_state'])
        dis_optim.load_state_dict(checkpoint['dis_optim_state'])
        epoch = checkpoint['epoch']
        print("!!!Loaded model states from disk!!!")
        print("Starting at epoch", epoch)
    except:
        gen.apply(weights_init)
        dis.apply(weights_init)
        epoch = 1
        print("!!!Initialized models from scratch!!!")

    onestep = OneStep(gen)
    
    
    print("Creating test audio...")
    stft = AudioToStftWithGradients.apply(EXAMPLE_RESULTS[2].float(), DIS_N_FFT, DIS_HOP_LEN)
    specgram = stft_to_specgram(stft)
    print(specgram.shape)
    stft = specgram_to_stft(specgram)
    audio = stft_to_audio(stft, DIS_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_SAVING)
    print(audio.shape)
    write_normalized_audio_to_disk(EXAMPLE_RESULTS[2].float(), './test1.wav')
    write_normalized_audio_to_disk(audio, "./test2.wav")
    
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(melspec.cpu().numpy(), ax=ax, sr=SAMPLE_RATE, hop_length=VIS_HOP_LEN)
    # fig.colorbar(img, ax=ax)
    # plt.show()

    # plt.figure()
    # librosa.display.waveplot(audio.cpu().numpy(), sr=SAMPLE_RATE)
    # plt.show()

    print("Initializing Visualizer...")
    pygame.init()
    ta_surface = pygame.display.set_mode((VIS_WIDTH, VIS_HEIGHT))
    ta_clk = pygame.time.Clock()

    if GEN_MODE in [5]:
        print("Creating CSound Interface...")
        result = G_csi.compile()
        if result != 0:
            raise RuntimeError("CSound compilation failed!")
        current_params = [0.]*N_PARAMS*TOTAL_PARAM_UPDATES

    #print_global_constants()
    print("Initialization complete! Starting...")
    time.sleep(5)

    i = curses.wrapper(train_until_interrupt, epoch, True)
    print("Generating...")
    start_time = time.time()
    with torch.no_grad():
        if GEN_MODE in [5]:
            params = onestep.generate_one_step()
            data = get_output_from_params(params, None)
        else:
            data = onestep.generate_one_step()
            if GEN_MODE == 4 and DIS_MODE == 2:
                data = MelToSTFTWithGradients.apply(data[:,:], N_GEN_MEL_CHANNELS)
                data = stft_to_audio(data, GEN_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_SAVING)
    end_time = round(time.time() - start_time, 2)
    print("Generated output in", end_time, "sec.")
    if G_csi:
        G_csi.stop()

    write_normalized_audio_to_disk(data, 'out1.wav')

    #plot_metrics(i, save_to_disk=True)

    print("Saving models...")
    save_states(i)

    print("Done!")
    dist.destroy_process_group()
    pygame.quit()

