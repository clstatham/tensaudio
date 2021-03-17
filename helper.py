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
    return tf.concat((tf.zeros([1]), tf.experimental.numpy.diff(p) / np.pi), axis=0) #(2.0*np.pi) * len(p)

#https://github.com/magenta/magenta/blob/c1340b2788af9bc193ef23e1ecec3fabf13d0a14/magenta/models/gansynth/lib/spectral_ops.py#L142
def diff(x, axis=-1):
    """Take the finite difference of a tensor along an axis.
    Args:
        x: Input tensor of any dimension.
        axis: Axis on which to take the finite difference.
    Returns:
        d: Tensor with size less than x by 1 along the difference dimension.
    Raises:
        ValueError: Axis out of range for tensor.
    """
    shape = x.get_shape()
    if axis >= len(shape):
        raise ValueError('Invalid axis index: %d for tensor with only %d axes.' %
                        (axis, len(shape)))

    begin_back = [0 for unused_s in range(len(shape))]
    begin_front = [0 for unused_s in range(len(shape))]
    begin_front[axis] = 1

    size = shape.as_list()
    size[axis] -= 1
    slice_front = tf.slice(x, begin_front, size)
    slice_back = tf.slice(x, begin_back, size)
    d = slice_front - slice_back
    return d

#https://github.com/magenta/magenta/blob/c1340b2788af9bc193ef23e1ecec3fabf13d0a14/magenta/models/gansynth/lib/spectral_ops.py#L172
def unwrap(p, discont=np.pi, axis=-1):
    """Unwrap a cyclical phase tensor.
    Args:
        p: Phase tensor.
        discont: Float, size of the cyclic discontinuity.
        axis: Axis of which to unwrap.
    Returns:
        unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff(p, axis=axis)
    ddmod = tf.math.mod(dd + np.pi, 2.0 * np.pi) - np.pi
    idx = tf.logical_and(tf.equal(ddmod, -np.pi), tf.greater(dd, 0))
    ddmod = tf.where(idx, tf.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    idx = tf.less(tf.abs(dd), discont)
    ddmod = tf.where(idx, tf.zeros_like(ddmod), dd)
    ph_cumsum = tf.cumsum(ph_correct, axis=axis)

    shape = p.get_shape().as_list()
    shape[axis] = 1
    ph_cumsum = tf.concat([tf.zeros(shape, dtype=p.dtype), ph_cumsum], axis=axis)
    unwrapped = p + ph_cumsum
    return unwrapped

#https://github.com/magenta/magenta/blob/c1340b2788af9bc193ef23e1ecec3fabf13d0a14/magenta/models/gansynth/lib/spectral_ops.py#L199
def gs_instantaneous_frequency(phase_angle, time_axis=-1, use_unwrap=True):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Take the finite difference of the phase. Pad with initial phase to keep the
    tensor the same size.
    Args:
        phase_angle: Tensor of angles in radians. [Batch, Time, Freqs]
        time_axis: Axis over which to unwrap and take finite difference.
        use_unwrap: True preserves original GANSynth behavior, whereas False will
            guard against loss of precision.
    Returns:
        dphase: Instantaneous frequency (derivative of phase). Same size as input.
    """
    if use_unwrap:
        # Can lead to loss of precision.
        phase_unwrapped = unwrap(phase_angle, axis=time_axis)
        dphase = diff(phase_unwrapped, axis=time_axis)
    else:
        # Keep dphase bounded. N.B. runs faster than a single mod-2pi expression.
        dphase = diff(phase_angle, axis=time_axis)
        dphase = tf.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
        dphase = tf.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)

    # Add an initial phase to dphase.
    size = phase_angle.get_shape().as_list()
    size[time_axis] = 1
    begin = [0 for unused_s in size]
    phase_slice = tf.slice(phase_angle, begin, size)
    dphase = tf.concat([phase_slice, dphase], axis=time_axis) / np.pi
    return dphase


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
    f = tf.squeeze(specgram[:,1])
    p = mel_to_linear(f)
    phase_angle = tf.cumsum(p * np.pi, -1) # convert from instantaneous frequency to instantaneous phase
    return polar_to_rect(mag, phase_angle)

def audio_to_specgram(audio):
    out = tf.TensorArray(tf.float32, size=tf.shape(audio)[0])
    for i, batch in enumerate(audio):
        ana = hilbert_from_scratch_tf(batch)
        a = tf.math.abs(ana)
        p = tf.math.atan2(tf.math.imag(ana), tf.math.real(ana))
        f = linear_to_mel(gs_instantaneous_frequency(p, use_unwrap=False))
        #f = inst_freq_pytorch_2(p, use_unwrap=False)
        out = out.write(i, tf.stack((a, f), axis=-1))
    return out.stack()

def specgram_to_audio(specgram):
    out = tf.TensorArray(tf.float32, size=tf.shape(specgram)[0])
    for i in range(tf.shape(specgram)[0]):
        fft = specgram_to_fft(specgram[i])
        invhilb = hilbert_from_scratch_tf(fft)
        out = out.write(i, tf.math.imag(invhilb))
    return out.stack()


def generate_input_noise(batch_size=BATCH_SIZE, noise_dims=TOTAL_SAMPLES_IN):
    if GEN_MODE in [10]:
        #return torch.randn(1, TOTAL_SAMPLES_IN, 1, requires_grad=True)
        raise NotImplementedError
    else:
        return tf.random.normal([batch_size, noise_dims], dtype=tf.float32)