import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow_transform as tft

import numpy as np
import scipy
import scipy.optimize

import librosa
import librosa.util
import soundfile
import matplotlib
import matplotlib.pyplot as plt

from global_constants import *

over_sample = 4
res_factor = 0.8
octaves = 6
notes_per_octave=10


# Plotting functions
cdict  = {'red':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'alpha':  ((0.0, 1.0, 1.0),
                   (1.0, 0.0, 0.0))
        }

my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
plt.register_cmap(cmap=my_mask)

def note_specgram(audio, ax, hop_length, peak=70.0):
    if hop_length >= 32:
        C = librosa.cqt(audio, sr=SAMPLE_RATE, hop_length=hop_length, 
                        bins_per_octave=int(notes_per_octave*over_sample), 
                        n_bins=int(octaves * notes_per_octave * over_sample),
                        filter_scale=res_factor, 
                        fmin=librosa.note_to_hz('C2'))
        mag, phase = librosa.core.magphase(C)
        phase_angle = np.angle(phase)
        phase_unwrapped = np.unwrap(phase_angle)
        dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
        dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
        mag = (librosa.amplitude_to_db(mag**2, amin=1e-13, top_db=peak) / peak) + 1
        ax.matshow(dphase[::-1, :], cmap=plt.cm.rainbow)
        ax.matshow(mag[::-1, :], cmap=my_mask)


def normalize_negone_one(x):
    # a = x - x.min(-1, keepdim=True)[0]
    # b = a / (a.max(-1, keepdim=True)[0] + EPSILON)
    # return 2 * b - 1
    return tft.scale_to_0_1(x) * 2 - 1
def normalize_zero_one(x):
    # a = x - x.min(-1, keepdim=True)[0]
    # b = a / (a.max(-1, keepdim=True)[0] + EPSILON)
    # return b
    return tft.scale_to_0_1(x)
    
def write_normalized_audio_to_disk(sig, fn):
    sig_numpy = librosa.util.normalize(sig)
    #scaled = np.int16(sig_numpy * 32767)
    scipy.io.wavfile.write(fn, SAMPLE_RATE, sig_numpy)

def polar_to_rect(mag, phase_angle):
    mag = tf.complex((mag, tf.zeros([1], dtype=mag.dtype)))
    phase = tf.complex(tf.cos(phase_angle), tf.sin(phase_angle))
    return mag * phase

# def stft_to_specgram(stft):
#     logmag = torch.abs(stft)
#     phase_angle = torch.angle(stft)
#     p = inst_freq_pytorch_2(phase_angle)
#     #p = phase_angle
#     return torch.stack((logmag, p), dim=0)

# def specgram_to_stft(specgram):
#     mag = specgram[0,:,:]
#     p = specgram[1,:,:]
#     phase_angle = torch.cumsum(p * np.pi, -2)
#     #phase_angle = p * np.pi
#     return polar_to_rect(mag, phase_angle)

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def mel_to_linear(m):
    return _MEL_BREAK_FREQUENCY_HERTZ * (
        tf.exp(m / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )
def linear_to_mel(f):
    return _MEL_HIGH_FREQUENCY_Q * tf.log(
        1.0 + (f / _MEL_BREAK_FREQUENCY_HERTZ))

# def hilbert_from_scratch_pytorch(u):
#     # N : fft length
#     # M : number of elements to zero out
#     # U : DFT of u
#     # v : IDFT of H(U)

#     N = len(u.view(-1))
#     # take forward Fourier transform
#     U = torch.fft.fft(u.view(-1))
#     M = N - N//2 - 1
#     # zero out negative frequency components
#     U[N//2+1:] = torch.zeros(M)
#     # double fft energy except @ DC0
#     U[1:N//2] = 2 * U[1:N//2]
#     # take inverse Fourier transform
#     v = torch.fft.ifft(U)
#     return v

def stftgram_to_stft(stftgram):
    mag = tf.squeeze(stftgram[0])
    p = tf.squeeze(stftgram[1])
    #phase_angle = torch.cumsum(p * np.pi, -1)
    phase_angle = p * np.pi
    return polar_to_rect(mag, phase_angle)

def specgram_to_fft(specgram):
    mag = tf.squeeze(stftgram[0])
    p = tf.squeeze(stftgram[1])
    #phase_angle = torch.cumsum(p * np.pi, -1)
    phase_angle = mel_to_linear(p * np.pi)
    return polar_to_rect(mag, phase_angle)

def audio_to_specgram(audio):
    out = []
    for batch in audio:
        ana = scipy.signal.hilbert(batch)
        mel_ana = ana
        a = tf.abs(mel_ana)
        p = linear_to_mel(tf.atan2(mel_ana.imag, mel_ana.real) / np.pi)
        #f = inst_freq_pytorch_2(p, use_unwrap=False)
        out.append(tf.stack((a, p), axis=0))
    return tf.stack(out)

def specgram_to_audio(specgram):
    out = []
    for batch in specgram:
        fft = specgram_to_fft(batch)
        invhilb = scipy.signal.hilbert(fft)
        out.append(tf.imag(invhilb))
    return tf.stack(out)
