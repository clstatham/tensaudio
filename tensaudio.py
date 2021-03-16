import collections
import curses
from threading import Thread, Lock
from queue import Queue
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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow_io as tfio
from matplotlib import animation, cm
from pygame.locals import *

#from csoundinterface import CSIRun, CsoundInterface, G_csi
from discriminator import *
from generator import *
from global_constants import *
from helper import *
from hilbert import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
devs = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devs[0], True)
plt.switch_backend('agg')

np.random.seed(int(round(time.time())))

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
    

def random_phase_shuffle(inp, chance, scale):
    if chance > 0 and chance < 1:
        spec = audio_to_specgram(inp)
        out = spec[:,:, 0]
        noise = normalize_negone_one(tf.random.normal((spec.shape[0], spec.shape[1]))) \
            * scale * tf.reduce_max(tf.abs(spec[:,:,1]))
        noise = K.dropout(noise, 1 - chance)
        new_phase = spec[:,:,1] + noise
        out = tf.stack([out, new_phase], axis=-1)
        return specgram_to_audio(out)
    else:
        return inp


global VIS_qlock

class TAMetricsPlotter():
    def __init__(self, grid_tick_interval):
        self.vis_queue = Queue()
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

    def handle_pygame_events(self):
        if not self.vis_queue.empty():
            metrics = {}
            for entry in range(self.vis_queue.qsize()):
                key, val = self.vis_queue.get()
                metrics[key] = val
            self.total_gen_losses.append(metrics['gen_loss'].numpy())
            self.total_dis_losses.append(metrics['dis_loss'].numpy())
            self.total_real_verdicts.append(metrics['real_verdict'].numpy())
            self.total_fake_verdicts.append(metrics['fake_verdict'].numpy())
            # if len(self.total_gen_losses) > self.graph_x_limit:
            #     self.total_gen_losses = self.total_gen_losses[1:]
            # if len(self.total_dis_losses) > self.graph_x_limit:
            #     self.total_dis_losses = self.total_dis_losses[1:]
            # if len(self.total_real_verdicts) > self.graph_x_limit:
            #     self.total_real_verdicts = self.total_real_verdicts[1:]
            # if len(self.total_fake_verdicts) > self.graph_x_limit:
            #     self.total_fake_verdicts = self.total_fake_verdicts[1:]
            
            try:
                self.current_output = metrics['test'].numpy()
                self.current_validation = metrics['validation'].numpy()
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
            if not self.paused:
                 next(self.plot_metrics(False))
            self.pg_clk.tick(60)
            pygame.display.flip()
        except:
            pass
        yield False

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

class TensAudio(keras.Model):
    def __init__(self):
        super().__init__()

        print("Creating Generator...")
        self.gen = TAGenerator()
        print("Creating Discriminator...")
        self.dis = TADiscriminator()
        
        
        self.validation_z = generate_input_noise()

    def compile(self):
        super().compile()
        self.gen_optim = tf.keras.optimizers.Adam(GEN_LR, BETA)
        self.dis_optim = tf.keras.optimizers.Adam(DIS_LR, BETA)
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.real_verdict_metric = keras.metrics.Mean(name="real_verdict")
        self.fake_verdict_metric = keras.metrics.Mean(name="fake_verdict")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def save_weights(self):
        self.gen.save_weights(os.path.join(MODEL_DIR, 'gen.ckpt'))
        self.dis.save_weights(os.path.join(MODEL_DIR, 'dis.ckpt'))
    def load_weights(self):
        self.gen.load_weights(os.path.join(MODEL_DIR, 'gen.ckpt'))
        self.dis.save_weights(os.path.join(MODEL_DIR, 'dis.ckpt'))

    #@tf.function
    def training_step(self, batch):
        batch_size = tf.shape(batch)[0]
        z = generate_input_noise()[:batch.shape[0]]
        gen_output = self.gen.gen_fn(z, training=True)
        combined = tf.concat([gen_output, batch], axis=0)
        combined_phase_shuffled = random_phase_shuffle(combined, PHASE_SHUFFLE_CHANCE, PHASE_SHUFFLE)
        dis_labels = tf.concat(
            [tf.zeros(tf.shape(batch)[0], 1), tf.ones(tf.shape(batch)[0], 1)], axis=0
        )
        dis_labels += tf.clip_by_value(0.05 * tf.random.uniform(tf.shape(dis_labels)), 0., 1.)

        with tf.GradientTape() as dis_tape:
            dis_output = self.dis(combined)
            dis_loss = keras.losses.binary_crossentropy(dis_labels, dis_output)
        dis_grads = dis_tape.gradient(dis_loss, self.dis.trainable_variables)
        self.dis_optim.apply_gradients(zip(dis_grads, self.dis.trainable_variables))

        gen_labels = tf.ones((tf.shape(batch)[0], 1))

        with tf.GradientTape() as gen_tape:
            dis_output_gen = self.dis(self.gen.gen_fn(z, training=True))
            gen_loss = keras.losses.binary_crossentropy(gen_labels, dis_output_gen)
        gen_grads = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        self.gen_optim.apply_gradients(zip(gen_grads, self.gen.trainable_variables))
        


        self.d_loss_metric.update_state(dis_loss)
        self.g_loss_metric.update_state(gen_loss)
        self.real_verdict_metric.update_state(tf.reduce_mean(dis_output[batch_size:], axis=0))
        self.fake_verdict_metric.update_state(tf.reduce_mean(dis_output[:batch_size], axis=0))
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "real_verdict": self.real_verdict_metric.result(),
            "fake_verdict": self.fake_verdict_metric.result(),
        }

        # VIS.put_queue(('gen_loss', gen_loss.detach().cpu()))
        # VIS.put_queue(('dis_loss', dis_loss.detach().cpu()))
        # VIS.put_queue(('real_verdict', dis_output_real.mean().detach().cpu()))
        # VIS.put_queue(('fake_verdict', dis_output_fake.mean().detach().cpu()))

    def train(self, ds, epochs=None):
        if epochs is None:
            epochs = 99999999999999999999999
        e = 0
        while e < epochs:
            start = time.time()
            print('='*80)
            print('Epoch', e, 'started.')
            for batch in ds:
                losses = self.training_step(batch)
                print('D loss:', round(losses['d_loss'].numpy(), 3), '\tG loss:', round(losses['g_loss'].numpy(), 3))
                VIS.put_queue(('gen_loss', losses['g_loss']))
                VIS.put_queue(('dis_loss', losses['d_loss']))
                VIS.put_queue(('real_verdict', losses['real_verdict']))
                VIS.put_queue(('fake_verdict', losses['fake_verdict']))
                next(VIS.handle_pygame_events())
            print()
            print('Epoch', e, 'finished in', round(time.time() - start, 2), 'seconds.')
            if e % SAVE_EVERY_EPOCH == 0:
                print('Generating progress report...')
                dirname = os.path.join(TRAINING_DIR, datetime.now().strftime("%d.%m.%Y"))
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                timestamp = datetime.now().strftime("%H.%M.%S")
                filename1 = os.path.join(dirname, timestamp+'_'+str(e)+"_progress"+"_EOE.wav")
                filename2 = os.path.join(dirname, timestamp+'_'+str(e)+"_training"+"_EOE.wav")
                prog = self.gen_progress()[0]
                trai = self.gen_random()[0]
                write_normalized_audio_to_disk(prog, filename1)
                write_normalized_audio_to_disk(trai, filename2)
                VIS.put_queue(('validation', prog))
                VIS.put_queue(('test', trai))
                #self.generate_progress_report(False, force_send=True)
                print('Progress report saved at', filename1, '.')
                print('Training report saved at', filename2, '.')
            print()
            e += 1
    
    def gen_random(self):
        return self.gen.gen_fn(generate_input_noise(), training=False)
    
    def gen_progress(self):
        return self.gen.gen_fn(self.validation_z, training=False)
            
    
    # def generate_progress_report(self, save=False, batch_idx=None, force_send=False):
    #     if not VIS.paused or force_send:
    #         gen_output_valid = self.gen_progress()
    #         for i, b in enumerate(gen_output_valid):
    #             if tf.reduce_max(tf.abs(b)) > 0:
    #                 VIS.put_queue(('validation', b))
    #                 break

    #         gen_output_test = self.gen_random()
    #         for i, b in enumerate(gen_output_test):
    #             if tf.reduce_max(tf.abs(b)) > 0:
    #                 VIS.put_queue(('test', b))
    #                 break

@tf.function
def load_sound(fn):
    def loader(_fn):
        data = tfio.audio.AudioIOTensor(_fn, dtype=tf.float32)
        sr = data.rate
        data = data.to_tensor()
        #data, sr = tfio.audio.decode_wav(f, dtype=tf.int32)
        sr = tf.cast(sr, tf.int64)
        if len(data.shape) > 1:
            data = tf.reshape(data, [-1])
        # sr_quotient = sr / SAMPLE_RATE
        # new_len = OUTPUT_DURATION * sr_quotient
        # new_samples = int(new_len * sr)
        # audio = data.reshape(-1)[:new_samples]
        audio_resampled = tf.cast(tfio.audio.resample(data, sr, SAMPLE_RATE), tf.float32)
        if len(audio_resampled) < TOTAL_SAMPLES_OUT:
            #print(tf.size(audio_resampled), tf.size(data))
            audio_resampled = tf.concat((audio_resampled, tf.zeros(TOTAL_SAMPLES_OUT-len(audio_resampled))), axis=0)
        return audio_resampled[:TOTAL_SAMPLES_OUT]
    out = tf.cast(loader(fn), tf.float32)
    # out = tf.ensure_shape(out, [TOTAL_SAMPLES_OUT])
    return out

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    global VIS, VIS_qlock
    print("TensAudio version 0.1 by https://github.com/clstatham")
    #print_global_constants()

    print("Creating Models...")
    model = TensAudio()
    ds = tfio.audio.AudioIODataset.list_files('C:/tensaudio_resources/piano/ta/*.wav', shuffle=True)
    ds = ds.map(load_sound, num_parallel_calls=tf.data.AUTOTUNE)
    ds = configure_for_performance(ds)

    print("Creating Test Audio...")
    inp = next(iter(ds))
    gram = audio_to_specgram(inp)
    #stft = audio_to_stftgram(inp, DIS_N_FFT, DIS_HOP_LEN)
    # print(stft.size())
    # print(gram[:,0].min(), gram[:,0].max())
    # print(gram[:,1].min(), gram[:,1].max())
    # print(gram[:,0].sum())
    # print(gram[:,1].sum())
    audio1 = specgram_to_audio(gram)
    #audio2 = stftgram_to_audio(stft, DIS_HOP_LEN)
    audio3 = random_phase_shuffle(inp, PHASE_SHUFFLE_CHANCE, PHASE_SHUFFLE)
    #audio4 = F.dropout(audio3, DIS_DROPOUT, training=True, inplace=False)
 
    write_normalized_audio_to_disk(inp[0], './original audio.wav')
    write_normalized_audio_to_disk(audio1[0], "./specgram.wav")
    # write_normalized_audio_to_disk(audio2.view(-1), "./stftgram.wav")
    write_normalized_audio_to_disk(audio3[0], "./specgram_shuffled.wav")
    # write_normalized_audio_to_disk(audio4.view(-1), "./specgram_shuffled_dropout.wav")

    # print("Real audio specgram:")
    # print("Mag min/max/mean:", gram[:,0].min(), gram[:,0].max(), gram[:,0].mean())
    # print("Phase min/max/mean:", gram[:,1].min(), gram[:,1].max(), gram[:,1].mean())

    print("Initializing 'lazy' modules...")

    

    #noise = generate_input_noise()#[-2:]
    model.compile()
    #model.build(list(tf.shape(inp)))
    #model.fit(ds.take(1), epochs=1)
    #model.training_step(inp)
    #model.summary()

    # gen_output = model.gen(noise)
    # dis_output_fake = model.dis(gen_output)
    # dis_output_real = model.dis(inp)
    # #print("Got", dis_output[0].item(), "and", dis_output[1].item(), "from the discriminator on initial pass.")
    # print("Got output size", tuple(dis_output_fake.shape), "from discriminator.")
    # print("Real verdict:", tf.reduce_mean(dis_output_real).numpy())
    # print("Fake verdict:", tf.reduce_mean(dis_output_fake).numpy())

    print("Initializing Visualizer...")
    VIS = TAMetricsPlotter(50*4)
    VIS_qlock = Lock()

    print("Attempting to load model weights...")
    try:
        model.load_weights()
    except Exception as e:
        print("Couldn't load weights:", e)

    print("Initialization complete! Starting...")
    time.sleep(1)
    
    #trainer.tune(model, data_mod)
    try:
        model.train(ds, epochs=None)
    except KeyboardInterrupt:
        print('Saving...')
        model.save_weights()
        print('Generating final progress report...')
        write_normalized_audio_to_disk(model.gen_progress()[0], './out.wav')

    print("Done!")

    pygame.quit()
