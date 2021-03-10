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
        return DataLoader(self.set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    def test_dataloader(self):
        return DataLoader(self.set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

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
    def __init__(self, grid_tick_interval=BATCH_SIZE):
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

        self.grid_tick_interval = 1045 // BATCH_SIZE # every epoch
        self.graph_x_limit = 20*self.grid_tick_interval

    def put_queue(self, val):
        VIS_qlock.acquire()
        self.vis_queue.put(val)
        VIS_qlock.release()

    def handle_pygame_events(self, step):
        if not self.vis_queue.empty():
            metrics = {}
            for entry in range(self.vis_queue.qsize()):
                key, val = self.vis_queue.get()
                metrics[key] = val
            self.total_gen_losses.append(metrics['gen_loss'].detach().cpu().item())
            self.total_dis_losses.append(metrics['dis_loss'].detach().cpu().item())
            self.total_real_verdicts.append(metrics['real_verdict'].detach().cpu().item())
            self.total_fake_verdicts.append(metrics['fake_verdict'].detach().cpu().item())
            if len(self.total_gen_losses) > self.graph_x_limit:
                self.total_gen_losses = self.total_gen_losses[1:]
            if len(self.total_dis_losses) > self.graph_x_limit:
                self.total_dis_losses = self.total_dis_losses[1:]
            if len(self.total_real_verdicts) > self.graph_x_limit:
                self.total_real_verdicts = self.total_real_verdicts[1:]
            if len(self.total_fake_verdicts) > self.graph_x_limit:
                self.total_fake_verdicts = self.total_fake_verdicts[1:]
            
            try:
                self.current_output = metrics['test'].detach().cpu().numpy()
                self.current_validation = metrics['validation'].detach().cpu().numpy()
                next(self.plot_metrics(True))
            except:
                pass
        try:
            pygame.event.pump()
            for ev in pygame.event.get():
                if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_x):
                    pygame.quit()
                    yield True
                elif ev.type == KEYDOWN:
                    if ev.key == K_p:
                        self.paused = not self.paused
                    if ev.key == K_SPACE:
                        next(self.plot_metrics(False))
            if step % VIS_UPDATE_INTERVAL == 0 and not self.paused:
                 next(self.plot_metrics(False))
            self.pg_clk.tick(60)
            pygame.display.flip()
        except:
            pass
        yield False

    @torch.no_grad()
    def plot_metrics(self, save_to_disk=False):
        self.pg_surface.fill((255,255,255))

        timestamp = str(datetime.now().strftime("%d.%m.%Y_%H.%M.%S"))
        dirname = os.path.join(PLOTS_DIR, timestamp)

        filename3 = str('LOSSES_'+timestamp+'.png')
        filename4 = str('SPECTROGRAMS_'+timestamp+'.png')

        if len(self.total_gen_losses) + len(self.total_dis_losses) + len(self.total_real_verdicts) + len(self.total_fake_verdicts) > 0:
            fig3, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=[VIS_WIDTH//100,VIS_HEIGHT//50], dpi=50, facecolor="white")
            fig3.subplots_adjust(hspace=0.1, wspace=0.1, left=0.05, bottom=0.05, right=0.95, top=0.95)

            ticks1 = np.arange(0, len(self.total_gen_losses), self.grid_tick_interval)
            ticks2 = np.arange(0, len(self.total_dis_losses), self.grid_tick_interval)
            ticks3 = np.arange(0, len(self.total_real_verdicts), self.grid_tick_interval)

            plt.title("Gen/Dis Losses " + timestamp)
            ax1.set_title("Gen Losses")
            ax1.set_facecolor("white")
            ax1.set_xticks(ticks1)
            ax1.grid(which='major', alpha=0.75)
            ax1.plot(range(len(self.total_gen_losses)), self.total_gen_losses, color="b")
            ax2.set_title("Dis Losses")
            ax2.set_facecolor("white")
            ax2.set_xticks(ticks2)
            ax2.grid(which='major', alpha=0.75)
            ax2.plot(range(len(self.total_dis_losses)), self.total_dis_losses, color="r")
            ax3.set_title("Real/Fake Verdicts")
            ax3.set_facecolor("white")
            ax3.set_xticks(ticks3)
            ax3.grid(which='major', alpha=0.75)
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
                spec_ax3.specgram(self.current_output, VIS_N_FFT, noverlap=VIS_N_FFT//2, mode='psd', cmap=plt.get_cmap('magma'))
                spec_ax4.specgram(self.current_validation, VIS_N_FFT, noverlap=VIS_N_FFT//2, mode='psd', cmap=plt.get_cmap('magma'))
                
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
        yield

global VIS

class TensAudio(pl.LightningModule):
    def __init__(self, 
        lr: float = LR,
        b1: float = BETA,
        b2: float = 0.999,
        gm: float = GEN_MOMENTUM,
        batch_size: int = BATCH_SIZE
    ):
        super().__init__()
        self.save_hyperparameters()

        print("Creating Generator...")
        self.gen = TAGenerator()
        print("Creating Discriminator...")
        self.dis = TADiscriminator()

        self.ready_to_stop = 0
        self.q = mp.Queue()
        self.log('early_stop_on', self.ready_to_stop, on_step=True, on_epoch=True, prog_bar=False)

        self.validation_z = generate_input_noise()
    
    def forward(self, z):
        return self.gen(z)
    
    def loss(self, label, val):
        return F.binary_cross_entropy(val, label)

    def training_step(self, batch, batch_idx, optimizer_idx):
        #if self.q.empty():
        #    self.ready_to_stop += 1
        #else:
        #    self.ready_to_stop = self.q.get()
        #self.log('early_stop_on', self.ready_to_stop, on_step=True, on_epoch=True, prog_bar=False)

        #self.zero_grad()

        if optimizer_idx == 0:
            z = generate_input_noise()
            gen_output = self(z)
            with torch.no_grad():
                dis_output_gen = self.dis(gen_output).requires_grad_(True)
            valid = torch.full_like(dis_output_gen, REAL_LABEL)
            valid.type_as(batch)
            gen_loss = self.loss(valid, dis_output_gen)
            VIS.put_queue(('gen_loss', gen_loss.detach().cpu()))
            output = OrderedDict({
                'loss': gen_loss,
            })
            self.log('gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True)
            
            return output
        
        if optimizer_idx == 1:
            gen_output = self.gen(generate_input_noise())
            dis_output_real = self.dis(batch)
            dis_output_fake = self.dis(gen_output.detach())
            valid = torch.full_like(dis_output_real, REAL_LABEL)
            valid.type_as(batch)
            fake = torch.full_like(dis_output_fake, FAKE_LABEL)
            fake.type_as(batch)
            real_loss = self.loss(valid, dis_output_real)
            fake_loss = self.loss(fake, dis_output_fake)
            dis_loss = (real_loss + fake_loss) / 2.
            output = OrderedDict({
                'loss': dis_loss,
            })
            VIS.put_queue(('dis_loss', dis_loss.detach().cpu()))
            VIS.put_queue(('real_verdict', dis_output_real.mean().detach().cpu()))
            VIS.put_queue(('fake_verdict', dis_output_fake.mean().detach().cpu()))
            self.log('dis_loss', dis_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('dis_output_real', dis_output_real.mean(), on_step=True, on_epoch=True, prog_bar=False)
            self.log('dis_output_fake', dis_output_fake.mean(), on_step=True, on_epoch=True, prog_bar=False)
            
            #self.log('early_stop_on', self.ready_to_stop, on_step=True, on_epoch=True, prog_bar=False)
            return output
        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        gm = self.hparams.gm
        print("Creating Optimizers...")
        gen_optim = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(b1, b2))
        dis_optim = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(b1, b2))
        return [gen_optim, dis_optim], []
    
    def generate_progress_report(self, batch_idx=None):
        gen_output_valid = self(self.validation_z)[0].cpu().detach()
        VIS.put_queue(('validation', gen_output_valid))
        gen_output_test = self(generate_input_noise())[0].cpu().detach()
        VIS.put_queue(('test', gen_output_test))

        dirname = os.path.join(TRAINING_DIR, datetime.now().strftime("%d.%m.%Y"))
        timestamp = datetime.now().strftime("%H.%M.%S")
        try:
            os.mkdir(dirname)
        except:
            pass
        if batch_idx is None:
            filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_progress"+"_EOE.wav")
        else:
            filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_progress"+"_"+str(batch_idx)+".wav")
        write_normalized_audio_to_disk(gen_output_valid, filename)

        dirname = os.path.join(TRAINING_DIR, datetime.now().strftime("%d.%m.%Y"))
        try:
            os.mkdir(dirname)
        except:
            pass
        if batch_idx is None:
            filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_training"+"_EOE.wav")
        else:
            filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_training"+"_"+str(batch_idx)+".wav")
        write_normalized_audio_to_disk(gen_output_test, filename)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % SAVE_EVERY_BATCHES == 0 and SAVE_EVERY_BATCHES > 0:
            self.generate_progress_report(batch_idx)

    # def on_train_epoch_end(self, *args, **kwargs):
    #     #self.generate_progress_report(None)
        
    #     output = OrderedDict({
    #         'early_stop_on': self.ready_to_stop,
    #     })
    #     self.log('early_stop_on', self.ready_to_stop, on_step=False, on_epoch=True, prog_bar=False)
    #     return output
    
    # def on_train_batch_start(self, *args, **kwargs):
    #     if self.ready_to_stop:
    #         return -1

global model

class TATrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        next(VIS.handle_pygame_events(self.counter))
        self.counter += 1
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)

if __name__ == "__main__":
    global VIS, model

    print("TensAudio version 0.1 by https://github.com/clstatham")
    
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
    data_mod = TASadDataModule(num_workers=1)
    data_mod.prepare_data()
    data_mod.setup()
    model = TensAudio()
    trainer = TATrainer(gpus=1, accelerator='dp', auto_scale_batch_size=False)#, limit_train_batches=8, max_epochs=4)

    print("Creating test audio...")
    inp = data_mod.set[1].unsqueeze(0)
    gram = audio_to_specgram(inp)
    stft = audio_to_stftgram(inp, DIS_N_FFT, DIS_HOP_LEN)
    # print(stft.size())
    # print(gram[:,0].min(), gram[:,0].max())
    # print(gram[:,1].min(), gram[:,1].max())
    # print(gram[:,0].sum())
    # print(gram[:,1].sum())
    audio1 = specgram_to_audio(gram)
    audio2 = stftgram_to_audio(stft, DIS_HOP_LEN)

    write_normalized_audio_to_disk(inp.view(-1), './original audio.wav')
    write_normalized_audio_to_disk(audio1.view(-1), "./specgram.wav")
    write_normalized_audio_to_disk(audio2.view(-1), "./stftgram.wav")

    print("Initializing 'lazy' modules...")
    noise = generate_input_noise()#[-2:]
    gen_output = model.gen(noise)
    dis_output = model.dis(gen_output)
    #print("Got", dis_output[0].item(), "and", dis_output[1].item(), "from the discriminator on initial pass.")
    print("Got output size", list(dis_output.shape), "from discriminator.")

    print("Initialization complete! Starting...")
    time.sleep(1)

    trainer.tune(model, data_mod)

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

