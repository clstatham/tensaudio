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
import tensorflow_gan as tfgan
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
tf.compat.v1.enable_eager_execution()
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
            self.total_gen_losses.append(metrics['gen_loss'])
            self.total_dis_losses.append(metrics['dis_loss'])
            self.total_real_verdicts.append(metrics['real_verdict'])
            self.total_fake_verdicts.append(metrics['fake_verdict'])
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

global VIS, GEN, DIS

#@tf.function(experimental_relax_shapes=True)
def generator(inp, mode):
    return GEN.call(inp, mode)

#@tf.function(experimental_relax_shapes=True)
def discriminator(inp):
    return DIS.call(inp)

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

def input_fn(mode, params):
    bs = params['batch_size']
    nd = params['noise_dims']
    split = 'train' if mode == tf.estimator.ModeKeys.TRAIN else 'test'
    shuffle = (mode == tf.estimator.ModeKeys.TRAIN)
    just_noise = (mode == tf.estimator.ModeKeys.PREDICT)
    noise_ds = (tf.data.Dataset.from_tensors(0).repeat()
              .map(lambda _: generate_input_noise(bs, nd)))
    if just_noise:
        return noise_ds
    
    
    return tf.data.Dataset.zip((noise_ds, ds))

def get_eval_metric_ops_fn(gan_model):
    real_data_logits = tf.reduce_mean(gan_model.discriminator_real_outputs)
    gen_data_logits = tf.reduce_mean(gan_model.discriminator_gen_outputs)
    return {
        'real_data_logits': tf.metrics.mean(real_data_logits),
        'gen_data_logits': tf.metrics.mean(gen_data_logits),
    }



"""
    #@tf.function
    

        # VIS.put_queue(('gen_loss', gen_loss.detach().cpu()))
        # VIS.put_queue(('dis_loss', dis_loss.detach().cpu()))
        # VIS.put_queue(('real_verdict', dis_output_real.mean().detach().cpu()))
        # VIS.put_queue(('fake_verdict', dis_output_fake.mean().detach().cpu()))
"""


if __name__ == "__main__":
    global VIS, VIS_qlock, GEN, DIS
    print("TensAudio version 0.1 by https://github.com/clstatham")
    #print_global_constants()

    print("Creating Dataset...")
    ds = tfio.audio.AudioIODataset.list_files('C:/tensaudio_resources/piano/ta/*.wav', shuffle=True)
    ds = ds.map(load_sound, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=100, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    VALIDATION_Z = generate_input_noise()

    print("Creating Generator...")
    GEN = TAGenerator()
    print("Creating Discriminator...")
    DIS = TADiscriminator()
    
    # gan_estimator = tfgan.estimator.gan_estimator.GANEstimator(
    #     model_dir=MODEL_DIR,
    #     generator_fn=generator,
    #     discriminator_fn=discriminator,
    #     generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    #     discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    #     params={'batch_size': BATCH_SIZE, 'noise_dims': TOTAL_SAMPLES_IN},
    #     generator_optimizer=tf.optimizers.Adam(GEN_LR, beta_1=BETA),
    #     discriminator_optimizer=tf.optimizers.Adam(DIS_LR, beta_1=BETA),
    #     get_eval_metric_ops_fn=get_eval_metric_ops_fn
    # )

    gen_optim = keras.optimizers.Adam(GEN_LR, beta_1=BETA)
    dis_optim = keras.optimizers.Adam(DIS_LR, beta_1=BETA)

    gradient_penalty_weight=10.0
    gradient_penalty_target=1.0

    def g_loss(fake_logits):
        return -tf.reduce_mean(fake_logits)
    
    def d_loss(real_logits, fake_logits):
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return fake_loss - real_loss

    def training_step(batch):
        batch_size = tf.shape(batch)[0]
        z = generate_input_noise()
        gen_output = generator(z, mode=tf.estimator.ModeKeys.TRAIN)
        combined = tf.concat([gen_output, batch], axis=0)
        #combined_phase_shuffled = random_phase_shuffle(combined, PHASE_SHUFFLE_CHANCE, PHASE_SHUFFLE)
        # dis_labels = tf.concat(
        #     [tf.zeros(tf.shape(batch)[0], 1), tf.ones(tf.shape(batch)[0], 1)], axis=0
        # )
        # dis_labels += tf.clip_by_value(0.05 * tf.random.uniform(tf.shape(dis_labels)), 0., 1.)

        with tf.GradientTape() as dis_tape:
            dis_output = discriminator(combined)
            dis_output_real = dis_output[batch_size:]
            dis_output_fake = dis_output[:batch_size]
            #dis_loss = tfgan.losses.losses_impl.wasserstein_discriminator_loss(dis_output_real, dis_output_fake)
            dis_loss = d_loss(dis_output_real, dis_output_fake)

            alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
            diff = gen_output - batch
            interpolated = batch + alpha * diff

            with tf.GradientTape() as gp_tape:
                gp_tape.watch(interpolated)
                # 1. Get the discriminator output for this interpolated image.
                pred = discriminator(interpolated)

            # 2. Calculate the gradients w.r.t to this interpolated image.
            grads = gp_tape.gradient(pred, [interpolated])[0]
            # 3. Calculate the norm of the gradients.
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + EPSILON)
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            dis_loss += gp * gradient_penalty_weight
        dis_grads = dis_tape.gradient(dis_loss, DIS.trainable_variables)
        dis_optim.apply_gradients(zip(dis_grads, DIS.trainable_variables))

        #gen_labels = tf.ones((tf.shape(batch)[0], 1))

        with tf.GradientTape() as gen_tape:
            dis_output_gen = discriminator(generator(z, mode=tf.estimator.ModeKeys.TRAIN))
            #gen_loss = tfgan.losses.losses_impl.wasserstein_generator_loss(dis_output_gen)
            gen_loss = g_loss(dis_output_gen)
        gen_grads = gen_tape.gradient(gen_loss, GEN.trainable_variables)
        gen_optim.apply_gradients(zip(gen_grads, GEN.trainable_variables))
        
        return {
            "d_loss": dis_loss.numpy(),
            "g_loss": gen_loss.numpy(),
            "real_verdict": tf.reduce_mean(dis_output_real, axis=0).numpy(),
            "fake_verdict": tf.reduce_mean(dis_output_fake, axis=0).numpy(),
        }

    def train(epochs=None):
        if epochs is None:
            epochs = 99999999999999999999999
        e = 0
        while e < epochs:
            start = time.time()
            print('='*80)
            print('Epoch', e, 'started.')
            for batch in ds:
                losses = training_step(batch)
                #gan_estimator.train(lambda: input_fn(tf.estimator.ModeKeys.TRAIN, gan_estimator.params), max_steps=4)
                #metrics = gan_estimator.evaluate(lambda: input_fn(tf.estimator.ModeKeys.EVAL, gan_estimator.params), steps=1)
                print("G Loss:", round(losses['g_loss'], 3), "\tD Loss:", round(losses['d_loss'], 3))
                VIS.put_queue(('gen_loss', losses['g_loss']))
                VIS.put_queue(('dis_loss', losses['d_loss']))
                VIS.put_queue(('real_verdict', losses['real_verdict']))
                VIS.put_queue(('fake_verdict', losses['fake_verdict']))
                next(VIS.handle_pygame_events())
            print()
            print('Epoch', e, 'finished in', round(time.time() - start, 2), 'seconds.')
            print()

            if e % SAVE_EVERY_EPOCH == 0:
                print('Generating progress report...')
                dirname = os.path.join(TRAINING_DIR, datetime.now().strftime("%d.%m.%Y"))
                if not os.path.isdir(dirname):
                    os.mkdir(dirname)
                timestamp = datetime.now().strftime("%H.%M.%S")
                filename1 = os.path.join(dirname, timestamp+'_'+str(e)+"_progress"+"_EOE.wav")
                filename2 = os.path.join(dirname, timestamp+'_'+str(e)+"_training"+"_EOE.wav")
                prog = generator(VALIDATION_Z, mode=None)[0]
                trai = generator(generate_input_noise(), mode=None)[0]
                write_normalized_audio_to_disk(prog, filename1)
                write_normalized_audio_to_disk(trai, filename2)
                VIS.put_queue(('validation', prog))
                VIS.put_queue(('test', trai))
                #self.generate_progress_report(False, force_send=True)
                print('Progress report saved at', filename1, '.')
                print('Training report saved at', filename2, '.')

            e += 1
            
        
        # def gen_random(self):
        #     return generator(generate_input_noise(), False)
        
        # def gen_progress(self):
        #     return generator(self.validation_z, False)
                
        
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

    

    print("Creating Test Audio...")
    inp = next(iter(ds))
    gram = audio_to_specgram(inp)
    # #stft = audio_to_stftgram(inp, DIS_N_FFT, DIS_HOP_LEN)
    # # print(stft.size())
    # # print(gram[:,0].min(), gram[:,0].max())
    # # print(gram[:,1].min(), gram[:,1].max())
    # # print(gram[:,0].sum())
    # # print(gram[:,1].sum())
    audio1 = specgram_to_audio(gram)
    # #audio2 = stftgram_to_audio(stft, DIS_HOP_LEN)
    # audio3 = random_phase_shuffle(inp, PHASE_SHUFFLE_CHANCE, PHASE_SHUFFLE)
    # #audio4 = F.dropout(audio3, DIS_DROPOUT, training=True, inplace=False)
 
    write_normalized_audio_to_disk(inp[0], './original audio.wav')
    write_normalized_audio_to_disk(audio1[0], "./specgram.wav")
    # # write_normalized_audio_to_disk(audio2.view(-1), "./stftgram.wav")
    # write_normalized_audio_to_disk(audio3[0], "./specgram_shuffled.wav")
    # # write_normalized_audio_to_disk(audio4.view(-1), "./specgram_shuffled_dropout.wav")

    # # print("Real audio specgram:")
    print("Mag min/max/mean:", tf.reduce_min(gram[:,:,0]), tf.reduce_max(gram[:,:,0]), tf.reduce_mean(gram[:,:,0]))
    print("Phase min/max/mean:", tf.reduce_min(gram[:,:,1]), tf.reduce_max(gram[:,:,1]), tf.reduce_mean(gram[:,:,1]))

    # print("Initializing 'lazy' modules...")


    print("Initializing Visualizer...")
    VIS = TAMetricsPlotter(25*3)
    VIS_qlock = Lock()

    # print("Attempting to load model weights...")
    # try:
    #     model.load_weights()
    # except Exception as e:
    #     print("Couldn't load weights:", e)

    print("Initialization complete! Starting...")
    time.sleep(1)
    
    try:
        train(epochs=None)
    except KeyboardInterrupt:
        print("Saving...")
        GEN.save_weights(os.path.join(MODEL_DIR, 'gen.ckpt'))
        DIS.save_weights(os.path.join(MODEL_DIR, 'dis.ckpt'))
        prog = generator(VALIDATION_Z, mode=None)[0]
        write_normalized_audio_to_disk(prog, './out.wav')
        pass

    print("Done!")

    pygame.quit()
