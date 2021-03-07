import collections
import curses
from threading import Thread, Lock
import os
import time
import glob
from datetime import datetime
from collections import OrderedDict

#import ctcsound
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
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LambdaCallback
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import animation, cm
from pygame.locals import *

#from csoundinterface import CSIRun, CsoundInterface, G_csi
from discriminator import TADiscriminator
from generator import TAGenerator, TAInstParamGenerator
from global_constants import *
from helper import *
from hilbert import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        a = torch.repeat_interleave(a, max(1,int(TOTAL_SAMPLES_OUT // a.view(-1).size(0))), dim=0)
        arr.append(a[:TOTAL_SAMPLES_OUT].float())
    return arr

class TASadDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.filenames = glob.glob(self.data_dir+str("/*.flac"))
        self.dims = (len(self.filenames))
    def __len__(self):
        return 1045
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx > len(self.filenames):
            raise ValueError("Index out of range", idx)
            return None
        
        fn = os.path.join(self.data_dir, self.filenames[idx])
        f = soundfile.SoundFile(fn, "r")
        data = torch.as_tensor(f.read()).requires_grad_(True)
        sr = torch.as_tensor(f.samplerate).unsqueeze(0)
        if len(data.shape) > 1 and data.shape[1] != 1:
            data = data.permute(1,0)[0].view(-1)
        sr_quotient = f.samplerate / SAMPLE_RATE
        new_len = OUTPUT_DURATION * sr_quotient
        new_samples = int(new_len * f.samplerate)
        audio = data.view(-1)[:new_samples]
        audio_resampled = torchaudio.transforms.Resample(sr.item(), SAMPLE_RATE)(audio.detach().clone())
        if audio_resampled.shape[0] < TOTAL_SAMPLES_OUT:
            audio_resampled = torch.cat((audio_resampled, torch.zeros(TOTAL_SAMPLES_OUT-audio_resampled.shape[0])))
        return audio_resampled.float()[:TOTAL_SAMPLES_OUT].cpu()

class TASadDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str = RESOURCES_DIR, batch_size:int = BATCH_SIZE, num_workers:int = 0):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "Musical Emotions Classification", "Sad")
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.set = TASadDataset(self.data_dir)
    def train_dataloader(self):
        return DataLoader(self.set, batch_size=self.batch_size, num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(self.set, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.set, batch_size=self.batch_size, num_workers=self.num_workers)

def select_input(idx):
    x = INPUTS[idx]
    if len(x) < TOTAL_SAMPLES_OUT:
        x = np.concatenate((x, [0]*(TOTAL_SAMPLES_OUT-len(x))))
    
    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_data(x)

def get_example(idx):
    x = EXAMPLES[idx]
    if len(x) < TOTAL_SAMPLES_OUT:
        x = np.concatenate((x, [0]*(TOTAL_SAMPLES_OUT-len(x))))

    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_data(x)
def get_random_example():
    return get_example(np.random.randint(len(EXAMPLES)))

def get_example_result(idx):
    x = EXAMPLE_RESULTS[idx]
    if len(x) < TOTAL_SAMPLES_OUT:
        x = torch.cat((x, torch.zeros(TOTAL_SAMPLES_OUT-len(x))))

    x = x[:TOTAL_SAMPLES_OUT]
    return normalize_data(x)
def get_random_example_result():
    return get_example_result(np.random.randint(len(EXAMPLE_RESULTS)))



class OneStep():
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def generate_one_step(self, noise=None):
        if noise is None:
            noise = generate_input_noise()
        self.generator.eval()
        with torch.no_grad():
            predicted_logits = self.generator(noise.detach().clone())
        self.generator.train()
        return torch.squeeze(predicted_logits)

def create_input():
        #x = get_random_example()
        y = normalize_audio(torch.as_tensor(next(ta_dataset_iterator)))
        #if INPUT_MODE == 'conv':
        #    y = torch.from_numpy(spectral_convolution(x, y))
        return y

def generate_input_noise():
    if GEN_MODE in [10]:
        return torch.randn(1, TOTAL_SAMPLES_IN, 1, requires_grad=True)
    elif GEN_MODE in [2]:
        return torch.randn(BATCH_SIZE, 2, TOTAL_SAMPLES_IN, requires_grad=True)
    elif GEN_MODE in [4]:
        return torch.randn(BATCH_SIZE, 1, N_GEN_MEL_CHANNELS, TOTAL_SAMPLES_IN, requires_grad=True)
    elif GEN_MODE in [5]:
        return torch.randn(BATCH_SIZE, 2, 2, TOTAL_SAMPLES_IN, requires_grad=True)
    elif GEN_MODE in [6]:
        return torch.randn(BATCH_SIZE, 2, TOTAL_SAMPLES_IN, requires_grad=True)
    else:
        return torch.randn(BATCH_SIZE, N_CHANNELS, TOTAL_SAMPLES_IN, requires_grad=True)
    
training_noise = ag.Variable(generate_input_noise(), requires_grad=True)
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
    audio = CSIRun.apply(params.clone())
    #noise = np.random.randn(len(audio)) * 0.0001
    # for i in range(len(audio)):
    #     if audio[i] == np.inf or audio[i] == np.nan:
    #         audio[i] = 0.00001
    return audio

class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer, pl_module)

global VIS_qlock

class TAMetricsPlotter():
    def __init__(self):
        self.vis_queue = mp.Queue()
        self.total_gen_losses = []
        self.total_dis_losses = []
        self.total_real_verdicts = []
        self.total_fake_verdicts = []
        self.current_test = None
        self.current_validation = None
        self.paused = True

        pygame.init()
        self.pg_surface = pygame.display.set_mode((VIS_WIDTH, VIS_HEIGHT))
        self.pg_clk = pygame.time.Clock()

    def put_queue(self, val):
        VIS_qlock.acquire()
        self.vis_queue.put(val)
        VIS_qlock.release()

    def handle_pygame_events(self, step):
        try:
            pygame.event.pump()
            for ev in pygame.event.get():
                if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_x):
                    pygame.quit()
                    return True
                elif ev.type == KEYDOWN:
                    if ev.key == K_p:
                        self.paused = not self.paused
                    if ev.key == K_SPACE:
                        self.plot_metrics(False)
            if step % VIS_UPDATE_INTERVAL == 0 and not self.paused:
                self.plot_metrics(False)
            if step % SAVE_EVERY_ITERS == 0 and SAVE_EVERY_ITERS > 0:
                self.plot_metrics(True)
            self.pg_clk.tick(60)
            pygame.display.flip()
        except:
            pass
        return False

    @torch.no_grad()
    def plot_metrics(self, save_to_disk=False):
        if not self.vis_queue.empty():
            self.pg_surface.fill((255,255,255))
            metrics = {}
            for entry in range(self.vis_queue.qsize()):
                key, val = self.vis_queue.get()
                metrics[key] = val
            self.total_gen_losses.append(metrics['gen_loss'].detach().cpu().item())
            self.total_dis_losses.append(metrics['dis_loss'].detach().cpu().item())
            self.total_real_verdicts.append(metrics['real_verdict'].detach().cpu().item())
            self.total_fake_verdicts.append(metrics['fake_verdict'].detach().cpu().item())
            try:
                self.current_output = metrics['test'].detach().cpu().numpy()
                self.current_validation = metrics['validation'].detach().cpu().numpy()
            except:
                pass

            timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
            dirname = os.path.join(PLOTS_DIR, timestamp)

            filename3 = str('LOSSES_'+timestamp+'.png')
            filename4 = str('SPECTROGRAMS_'+timestamp+'.png')

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

            if len(self.total_gen_losses) + len(self.total_dis_losses) + len(self.total_real_verdicts) + len(self.total_fake_verdicts) > 0:
                fig3, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=[VIS_WIDTH//100,VIS_HEIGHT//50], dpi=50, facecolor="white")
                fig3.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, bottom=0.05, right=0.95, top=0.95)
                plt.title("Gen/Dis Losses " + timestamp)
                ax1.set_title("Gen Losses")
                ax1.set_facecolor("white")
                ax1.plot(range(len(self.total_gen_losses)), self.total_gen_losses, color="b")
                ax2.set_title("Dis Losses")
                ax2.set_facecolor("white")
                ax2.plot(range(len(self.total_dis_losses)), self.total_dis_losses, color="r")
                ax3.set_title("Real/Fake Verdicts")
                ax3.set_facecolor("white")
                ax3.plot(range(len(self.total_real_verdicts)), self.total_real_verdicts, label="Real", color="g")
                ax3.plot(range(len(self.total_fake_verdicts)), self.total_fake_verdicts, label="Fake", color="m")
                ax3.legend()
                canvas = agg.FigureCanvasAgg(fig3)
                canvas.draw()
                renderer = canvas.get_renderer()
                plots_img = renderer.tostring_rgb()
                
                size_plots = canvas.get_width_height()
                

                surf_plots = pygame.image.fromstring(plots_img, size_plots, "RGB")
                self.pg_surface.blit(surf_plots, (0,0))
                #spec_out = librosa.feature.melspectrogram(current_output, SAMPLE_RATE, n_fft=N_FFT, hop_length=64)
                #librosa.display.specshow(librosa.power_to_db(spec_out, ref=np.max), y_axis='mel', fmax=SAMPLE_RATE/2, x_axis='time')
                #out_spec_fig = plt.figure(figsize=[VIS_WIDTH//100, VIS_HEIGHT//50], dpi=50)
                if self.current_validation is not None and self.current_output is not None:
                    out_spec_fig, (spec_ax3, spec_ax4) = plt.subplots(2, 1, figsize=[VIS_WIDTH//100, VIS_HEIGHT//50], dpi=50, facecolor="white")
                    out_spec_fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, bottom=0.05, right=0.95, top=0.95)

                    spec_ax3.set_title('Output')
                    spec_ax4.set_title('Progress')
                    spec_ax3.specgram(self.current_output, VIS_N_FFT, noverlap=VIS_N_FFT//8, mode='psd', cmap=plt.get_cmap('magma'))
                    spec_ax4.specgram(self.current_validation, VIS_N_FFT, noverlap=VIS_N_FFT//8, mode='psd', cmap=plt.get_cmap('magma'))
                    
                    canvas = agg.FigureCanvasAgg(out_spec_fig)
                    canvas.draw()
                    renderer = canvas.get_renderer()
                    out_spec_img = renderer.tostring_rgb()
                    size_out_spec_img = canvas.get_width_height()
                    plt.close()
                    out_spec_surf = pygame.image.fromstring(out_spec_img, size_out_spec_img, "RGB")
                    self.pg_surface.blit(out_spec_surf, (VIS_WIDTH//2, 0))
                
                plt.close()
                pygame.display.flip()

                # if save_to_disk:
                #     os.mkdir(dirname)
                #     fig3.savefig(os.path.join(dirname, filename3))
                #     out_spec_fig.savefig(os.path.join(dirname, filename4))

global VIS

class TensAudio(pl.LightningModule):
    def __init__(self, 
        latent_dim: int = 100,
        lr: float = LR,
        b1: float = BETA,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE
    ):
        super().__init__()
        self.save_hyperparameters()

        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        # dist.init_process_group('gloo', init_method="file:///D:/tensaudio/filestore", world_size=1, rank=0)

        print("Creating Generator...")
        self.gen = TAGenerator()
        #self.gen.share_memory()
        # gen_ddp = DDP(gen)
        print("Creating Discriminator...")
        self.dis = TADiscriminator()
        #self.dis.share_memory()
        # dis_ddp = DDP(dis)

        self.ready_to_stop = 0
        self.q = mp.Queue()
        self.log('early_stop_on', self.ready_to_stop, on_step=True, on_epoch=True, prog_bar=False)

        print("Initializing Visualizer...")
        self.vis = TAMetricsPlotter()

        self.validation_z = generate_input_noise()
    
    def forward(self, z):
        return self.gen(z)
    
    def loss(self, label, val):
        return F.binary_cross_entropy(val, label)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data = batch
        z = generate_input_noise()
        
        if self.q.empty():
            self.ready_to_stop += 1
        else:
            self.ready_to_stop = self.q.get()
        self.log('early_stop_on', self.ready_to_stop, on_step=True, on_epoch=True, prog_bar=False)

        if optimizer_idx == 0:
            self.gen_output = self(z)
            dis_output_gen = self.dis(self(z))
            valid = torch.full_like(dis_output_gen, REAL_LABEL)
            valid.type_as(data)
            gen_loss = self.loss(valid, dis_output_gen)
            VIS.put_queue(('gen_loss', gen_loss.detach().cpu()))
            output = OrderedDict({
                'loss': gen_loss,
                'early_stop_on': self.ready_to_stop,
            })
            self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True)
            
            return output
        
        if optimizer_idx == 1:
            dis_output_real = self.dis(data)
            valid = torch.full_like(dis_output_real, REAL_LABEL)
            valid.type_as(data)

            real_loss = self.loss(valid, dis_output_real)

            dis_output_fake = self.dis(self.gen(z).detach())
            fake = torch.full_like(dis_output_fake, FAKE_LABEL)
            fake_loss = self.loss(fake, dis_output_fake)

            dis_loss = (real_loss + fake_loss) / 2
            output = OrderedDict({
                'loss': dis_loss,
                'early_stop_on': self.ready_to_stop,
            })
            VIS.put_queue(('dis_loss', dis_loss.detach().cpu()))
            VIS.put_queue(('real_verdict', dis_output_real.detach().cpu()))
            VIS.put_queue(('fake_verdict', dis_output_fake.detach().cpu()))
            self.log('dis_loss', dis_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('early_stop_on', self.ready_to_stop, on_step=True, on_epoch=True, prog_bar=False)
            return output
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        print("Creating Optimizers...")
        gen_optim = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        dis_optim = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(b1, b2))
        return [gen_optim, dis_optim], []
    
    def on_train_epoch_end(self, *args, **kwargs):
        gen_output = self(self.validation_z)
        VIS.put_queue(('validation', gen_output.detach().cpu()))
        gen_output = self(generate_input_noise())
        VIS.put_queue(('test', gen_output.detach().cpu()))
        
        """
        dirname = os.path.join(TRAINING_DIR, datetime.now().strftime("%d.%m.%Y"))
        try:
            os.mkdir(dirname)
        except:
            pass
        write_normalized_audio_to_disk(
            gen_output,
            os.path.join(dirname,
                "progress_"+datetime.now().strftime("%H.%M.%S")+'_'+str(self.current_epoch)+".wav"))

        dirname = os.path.join(TRAINING_DIR, datetime.now().strftime("%d.%m.%Y"))
        try:
            os.mkdir(dirname)
        except:
            pass
        write_normalized_audio_to_disk(
            gen_output,
            os.path.join(dirname,
                "training_"+datetime.now().strftime("%H.%M.%S")+'_'+str(self.current_epoch)+".wav"))
        """
        output = OrderedDict({
            'early_stop_on': self.ready_to_stop,
        })
        self.log('early_stop_on', self.ready_to_stop, on_step=False, on_epoch=True, prog_bar=False)
        return output
    
    # def on_train_batch_start(self, *args, **kwargs):
    #     if self.ready_to_stop:
    #         return -1


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
    label = torch.full(verdict.shape, REAL_LABEL).to(torch.float)
    dis_loss_real = dis.criterion(label, verdict)
    dis_loss_real.backward()
    real_verdict = verdict.item()

    if GEN_MODE in [10]:
        noise = generate_input_noise()
        params = gen(noise)
        audio = get_output_from_params(params, window)
        fake = torch.squeeze(audio).float()
    else:
        noise = generate_input_noise()
        fake = gen(noise)
    
    verdict = dis(fake.detach()).view(-1)
    fake_verdict = verdict.item()
    if torch.isnan(verdict):
        print("foo")
        pass
    label = torch.full(verdict.shape, FAKE_LABEL).to(torch.float)
    dis_loss_fake = dis.criterion(label, verdict)
    dis_loss_fake.backward()
    dis_loss = dis_loss_real + dis_loss_fake
    #dis_loss.backward()
    dis_optim.step()

    gen.zero_grad()
    verdict = dis(fake).view(-1)
    
    label = torch.full(verdict.shape, REAL_LABEL).to(torch.float)
    #label = torch.full(y3.shape, REAL_LABEL).to(torch.float)
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
        z = torch.autograd.Variable(create_input(), requires_grad=True)
        begin_time = time.time()
        real, fake, gen_loss, dis_loss = run_models(window, z)
        if current_view_mode == 1:
            current_output = fake.detach().clone().cpu().numpy()
        elif current_view_mode in [2,3]:
            current_output = onestep.generate_one_step(training_noise).detach().clone().cpu().numpy()
        current_example = z.detach().clone().cpu().numpy()
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
    if GEN_MODE in [10]:
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

global model

class TATrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if VIS.handle_pygame_events(self.counter):
            VIS_qlock.acquire()
            model.q.put(-1)
            VIS_qlock.release()
        self.counter += 1
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)

if __name__ == "__main__":
    global VIS, model

    print("TensAudio version 0.1 by https://github.com/clstatham")

    # print("Creating test audio...")
    # inp = create_input().float()
    # gram = audio_to_specgram(inp)
    # print(gram[0].max())
    # print(gram[1].max())
    # audio = specgram_to_audio(gram)

    # write_normalized_audio_to_disk(inp, './original audio.wav')
    # write_normalized_audio_to_disk(audio, "./specgram.wav")
    
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(melspec.cpu().numpy(), ax=ax, sr=SAMPLE_RATE, hop_length=VIS_HOP_LEN)
    # fig.colorbar(img, ax=ax)
    # plt.show()

    # plt.figure()
    # librosa.display.waveplot(audio.cpu().numpy(), sr=SAMPLE_RATE)
    # plt.show()

    if GEN_MODE in [10]:
        print("Creating CSound Interface...")
        result = G_csi.compile()
        if result != 0:
            raise RuntimeError("CSound compilation failed!")
        current_params = [0.]*N_PARAMS*TOTAL_PARAM_UPDATES

    #print_global_constants()
    
    print("Initializing Visualizer...")
    VIS = TAMetricsPlotter()
    VIS_qlock = Lock()

    print("Creating Models...")

    #i = curses.wrapper(train_until_interrupt, epoch, True)
    stopper = MyEarlyStopping('early_stop_on_step',patience=0, strict=True, mode='max')
    data_mod = TASadDataModule(num_workers=0)
    data_mod.prepare_data()
    data_mod.setup()
    model = TensAudio()
    trainer = TATrainer(gpus=1, accelerator='dp')

    print("Initialization complete! Starting...")
    time.sleep(1)

    #trainer.tune(model, data_mod)

    trainer.fit(model, data_mod)

    #print("Done!")

    # start_time = time.time()
    # with torch.no_grad():
    #     if GEN_MODE in [10]:
    #         params = onestep.generate_one_step()
    #         data = get_output_from_params(params, None)
    #     else:
    #         data = onestep.generate_one_step()
    #         if GEN_MODE == 4 and DIS_MODE == 2:
    #             data = MelToSTFTWithGradients.apply(data[:,:], N_GEN_MEL_CHANNELS)
    #             data = stft_to_audio(data, GEN_HOP_LEN, GRIFFIN_LIM_MAX_ITERS_SAVING)
    # end_time = round(time.time() - start_time, 2)
    # print("Generated output in", end_time, "sec.")
    # if G_csi:
    #     G_csi.stop()

    # write_normalized_audio_to_disk(data, 'out1.wav')

    #plot_metrics(i, save_to_disk=True)

    #print("Saving models...")
    #save_states(i)

    #dist.destroy_process_group()
    #pygame.quit()

