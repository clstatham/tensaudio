import collections
import curses
from threading import Thread, Lock
import os
import time
import glob
from datetime import datetime
from collections import OrderedDict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds

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
from matplotlib import animation, cm
from pygame.locals import *

#from csoundinterface import CSIRun, CsoundInterface, G_csi
from discriminator import TADiscriminator
from generator import TAGenerator
from global_constants import *
from helper import *
from hilbert import *

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
plt.switch_backend('agg')

np.random.seed(int(round(time.time())))

def generate_input_noise():
    if GEN_MODE in [10]:
        raise NotImplementedError
    else:
        return tf.random.normal([BATCH_SIZE, 1, TOTAL_SAMPLES_IN])
    


# def random_phase_shuffle(inp, chance, scale):
#     out = inp.clone()
#     second_half = inp.shape[1] // 2
#     noise = normalize_negone_one(torch.randn((inp.shape[0], inp.shape[1]-second_half, *inp.shape[2:])).to(inp.device)) \
#         * scale * inp[:,second_half:].abs().max()
#     noise = F.dropout(noise, 1 - chance, True, False)
#     out[:,second_half:] = inp[:,second_half:] + noise
#     return out
"""
global VIS_qlock

class TAMetricsPlotter():
    def __init__(self, grid_tick_interval):
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

        self.start_time = datetime.now().strftime("%d.%m.%Y_%H.%M")

        #self.grid_tick_interval = 1045 // BATCH_SIZE # every epoch
        self.grid_tick_interval = grid_tick_interval
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
            # if len(self.total_gen_losses) > self.graph_x_limit:
            #     self.total_gen_losses = self.total_gen_losses[1:]
            # if len(self.total_dis_losses) > self.graph_x_limit:
            #     self.total_dis_losses = self.total_dis_losses[1:]
            # if len(self.total_real_verdicts) > self.graph_x_limit:
            #     self.total_real_verdicts = self.total_real_verdicts[1:]
            # if len(self.total_fake_verdicts) > self.graph_x_limit:
            #     self.total_fake_verdicts = self.total_fake_verdicts[1:]
            
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

        timestamp = str(self.start_time)
        dirname = os.path.join(PLOTS_DIR, timestamp)

        filename3 = str(str(datetime.now().strftime("%H.%M.%S"))+'_LOSSES.png')
        filename4 = str(str(datetime.now().strftime("%H.%M.%S"))+'_SPECTROGRAMS.png')

        if save_to_disk:
            try:
                os.mkdir(dirname)
            except:
                pass

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

            if save_to_disk:
                fig3.savefig(os.path.join(dirname, filename3))
            plt.close()
        
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
            
            out_spec_surf = pygame.image.fromstring(out_spec_img, size_out_spec_img, "RGB")
            self.pg_surface.blit(out_spec_surf, (VIS_WIDTH//2, 0))

            if save_to_disk:
                out_spec_fig.savefig(os.path.join(dirname, filename4))
            plt.close()
        
        pygame.display.flip()

        
            
            
        yield

global VIS
"""

class TensAudio():
    def __init__(self):
        super().__init__()

        print("Creating Generator...")
        self.gen = TAGenerator()
        print("Creating Discriminator...")
        self.dis = TADiscriminator()

        self.dis_optim = keras.optimizers.Adam(DIS_LR, beta_1=BETA)
        self.gen_optim = keras.optimizers.Adam(GEN_LR, beta_1=BETA)

        self.validation_z = generate_input_noise()

    def call(self, z):
        return self.gen(z)
    
    def loss(self, label, val):
        return keras.losses.binary_crossentropy(val, label)

    @tf.function
    def training_step(self, batch):
        z = generate_input_noise()
        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            gen_output = self.gen(z, training=True)

            dis_output_gen = self.dis(gen_output, training=True)
            dis_output_real = self.dis(batch, training=True)

            gen_loss = self.loss(tf.ones_like(dis_output_gen), dis_output_gen)
            dis_loss_real = self.loss(tf.ones_like(dis_output_real), dis_output_real)
            dis_loss_fake = self.loss(tf.zeros_like(dis_output_gen), dis_output_gen)
            dis_loss = (dis_loss_real + dis_loss_fake) / 2.
        
        gen_grads = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        dis_grads = dis_tape.gradient(dis_loss, self.dis.trainable_variables)
        self.gen_optim.apply_gradients(zip(gen_grads, self.gen.trainable_variables))
        self.dis_optim.apply_gradients(zip(dis_grads, self.dis.trainable_variables))

        # VIS.put_queue(('gen_loss', gen_loss.detach().cpu()))
        # VIS.put_queue(('dis_loss', dis_loss.detach().cpu()))
        # VIS.put_queue(('real_verdict', dis_output_real.mean().detach().cpu()))
        # VIS.put_queue(('fake_verdict', dis_output_fake.mean().detach().cpu()))
    
    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for batch in dataset:
                self.training_step(batch)
        

    
    # def generate_progress_report(self, save=False, batch_idx=None, force_send=False):
    #     v, t = None, None
    #     if not VIS.paused or force_send:
    #         gen_output_valid = self(self.validation_z).cpu().detach()
    #         for i, b in enumerate(gen_output_valid):
    #             if b.abs().max() > 0:
    #                 v = b
    #                 VIS.put_queue(('validation', b))
    #                 break
    #             # else:
    #             #     print("Skipping validation mini-batch of zeros", i, ".")
            
    #         gen_output_test = self(generate_input_noise()).cpu().detach()
    #         for i, b in enumerate(gen_output_test):
    #             if b.abs().max() > 0:
    #                 t = b
    #                 VIS.put_queue(('test', b))
    #                 break
    #             # else:
    #             #     print("Skipping testing mini-batch of zeros", i, ".")
        
    #     if save:
    #         if VIS.paused:
    #             gen_output_valid = self(self.validation_z).cpu().detach()
    #             for b in gen_output_valid:
    #                 if b.abs().max() > 0:
    #                     v = b
    #                     break
                
    #             gen_output_test = self(generate_input_noise()).cpu().detach()
    #             for b in gen_output_test:
    #                 if b.abs().max() > 0:
    #                     t = b
    #                     break
    #         dirname = os.path.join(TRAINING_DIR, datetime.now().strftime("%d.%m.%Y"))
    #         timestamp = datetime.now().strftime("%H.%M.%S")
    #         if v is not None:
    #             try:
    #                 os.mkdir(dirname)
    #             except:
    #                 pass
    #             if batch_idx is None:
    #                 filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_progress"+"_EOE.wav")
    #             else:
    #                 filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_progress"+"_"+str(batch_idx)+".wav")
    #             write_normalized_audio_to_disk(v, filename)
    #         if t is not None:
    #             try:
    #                 os.mkdir(dirname)
    #             except:
    #                 pass
    #             if batch_idx is None:
    #                 filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_training"+"_EOE.wav")
    #             else:
    #                 filename = os.path.join(dirname, timestamp+'_'+str(self.current_epoch)+"_training"+"_"+str(batch_idx)+".wav")
    #             write_normalized_audio_to_disk(t, filename)

    # def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
    #     if batch_idx % SAVE_EVERY_BATCHES == 0 and SAVE_EVERY_BATCHES > 0:
    #         self.generate_progress_report(False, batch_idx)
    #     else:
    #         self.generate_progress_report(False, batch_idx)

    # def on_train_epoch_end(self, *args, **kwargs):
    #     if self.current_epoch % SAVE_EVERY_EPOCH == 0 and SAVE_EVERY_EPOCH > 0:
    #         self.generate_progress_report(True, None, True)
    
    # def on_train_batch_start(self, *args, **kwargs):
    #     if self.ready_to_stop:
    #         return -1

if __name__ == "__main__":
    print("TensAudio version 0.2 by https://github.com/clstatham")

    print("Creating Models...")
    model = TensAudio()
    ds = tf.data.Dataset.list_files('C:/tensaudio_resources/piano/ta/*.wav', shuffle=True)

    print("Creating Test Audio...")
    inp = next(ds.as_numpy_iterator())
    gram = audio_to_specgram(inp)
    #stft = audio_to_stftgram(inp, DIS_N_FFT, DIS_HOP_LEN)
    # print(stft.size())
    # print(gram[:,0].min(), gram[:,0].max())
    # print(gram[:,1].min(), gram[:,1].max())
    # print(gram[:,0].sum())
    # print(gram[:,1].sum())
    audio1 = specgram_to_audio(gram)
    #audio2 = stftgram_to_audio(stft, DIS_HOP_LEN)
    #audio3 = specgram_to_audio(random_phase_shuffle(audio_to_specgram(inp), PHASE_SHUFFLE_CHANCE, PHASE_SHUFFLE))
    #audio4 = F.dropout(audio3, DIS_DROPOUT, training=True, inplace=False)
 
    write_normalized_audio_to_disk(inp.view(-1), './original audio.wav')
    write_normalized_audio_to_disk(audio1.view(-1), "./specgram.wav")
    # write_normalized_audio_to_disk(audio2.view(-1), "./stftgram.wav")
    # write_normalized_audio_to_disk(audio3.view(-1), "./specgram_shuffled.wav")
    # write_normalized_audio_to_disk(audio4.view(-1), "./specgram_shuffled_dropout.wav")

    print("Real audio specgram:")
    print("Mag min/max/mean:", gram[:,0].min(), gram[:,0].max(), gram[:,0].mean())
    print("Phase min/max/mean:", gram[:,1].min(), gram[:,1].max(), gram[:,1].mean())

    print("Initializing modules...")
    noise = generate_input_noise()#[-2:]
    gen_output = model.gen(noise)
    dis_output_fake = model.dis(gen_output)
    dis_output_real = model.dis(inp)
    #print("Got", dis_output[0].item(), "and", dis_output[1].item(), "from the discriminator on initial pass.")
    print("Got output size", tuple(dis_output_fake.shape), "from discriminator.")
    print("Real verdict:", dis_output_real.mean().item())
    print("Fake verdict:", dis_output_fake.mean().item())

    # print("Initializing Visualizer...")
    # VIS = TAMetricsPlotter(10 * int(np.ceil(len(data_mod.set) / BATCH_SIZE)))
    # VIS_qlock = Lock()

    print("Initialization complete! Starting...")
    time.sleep(1)

    #trainer.tune(model, data_mod)

    trainer.fit(model, data_mod)

    print("Done!")

    pygame.quit()
