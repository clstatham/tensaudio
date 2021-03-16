import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

import numpy as np
import scipy

import librosa
import librosa.util
import soundfile
import matplotlib
import matplotlib.pyplot as plt

from global_constants import *

def normalize_audio(audio):
    return F.normalize(audio.flatten(), dim=0)
def normalize_data(tensor):
    return tensor/(tensor.abs().max() + EPSILON)
def normalize_zero_one(x):
    a = x - tf.reduce_min(x, axis=-1, keepdims=True)
    b = a / (tf.reduce_max(a, axis=-1, keepdims=True) + EPSILON)
    return b
def normalize_negone_one(x):
    b = normalize_zero_one(x)
    return 2 * b - 1
    
def write_normalized_audio_to_disk(sig, fn):
    sig_numpy = librosa.util.normalize(sig.numpy().astype(np.float32))
    #scaled = np.int16(sig_numpy * 32767)
    scipy.io.wavfile.write(fn, SAMPLE_RATE, sig_numpy)

def inst_freq(p):
    return np.diff(p) / (2.0*np.pi) * len(p)

def polar_to_rect(mag, phase_angle):
    mag = tf.complex(mag, tf.zeros([1], dtype=mag.dtype))
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
        tf.math.exp(m / _MEL_HIGH_FREQUENCY_Q) - 1.0
    )
def linear_to_mel(f):
    return _MEL_HIGH_FREQUENCY_Q * tf.math.log(
        1.0 + (f / _MEL_BREAK_FREQUENCY_HERTZ))

def hilbert_from_scratch_tf(_u):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)
    u = tf.cast(tf.reshape(_u, [-1]), dtype=tf.complex64)
    N = tf.size(u)
    # take forward Fourier transform
    U = tf.signal.fft(u)
    M = N - N//2 - 1
    # zero out negative frequency components
    #U[N//2+1:] = tf.zeros(M)
    U_2ndhalf = tf.zeros(N//2+1, dtype=tf.complex64)
    # double fft energy except @ DC0
    #U[1:N//2] = 2 * U[1:N//2]
    U_1sthalf = 2 * U[1:N//2]
    U = tf.concat([U_1sthalf, U_2ndhalf], axis=0)
    # take inverse Fourier transform
    v = tf.signal.ifft(U)
    return v

def specgram_to_fft(specgram):
    mag = tf.squeeze(specgram[:,0])
    p = tf.squeeze(specgram[:,1])
    #phase_angle = torch.cumsum(p * np.pi, -1)
    phase_angle = mel_to_linear(p * np.pi)
    return polar_to_rect(mag, phase_angle)

def audio_to_specgram(audio):
    out = tf.TensorArray(tf.float32, size=tf.shape(audio)[0])
    for i, batch in enumerate(audio):
        ana = hilbert_from_scratch_tf(batch)
        a = tf.math.abs(ana)
        p = linear_to_mel(tf.math.atan2(tf.math.imag(ana), tf.math.real(ana)) / np.pi)
        #f = inst_freq_pytorch_2(p, use_unwrap=False)
        out = out.write(i, tf.stack((a, p), axis=-1))
    return out.stack()

def specgram_to_audio(specgram):
    out = tf.TensorArray(tf.float32, size=tf.shape(specgram)[0])
    for i in range(tf.shape(specgram)[0]):
        fft = specgram_to_fft(specgram[i])
        invhilb = hilbert_from_scratch_tf(fft)
        out = out.write(i, tf.math.imag(invhilb))
    return out.stack()


def generate_input_noise():
    if GEN_MODE in [10]:
        #return torch.randn(1, TOTAL_SAMPLES_IN, 1, requires_grad=True)
        raise NotImplementedError
    else:
        return tf.random.normal([BATCH_SIZE, TOTAL_SAMPLES_IN], dtype=tf.float32)