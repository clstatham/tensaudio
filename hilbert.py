import six
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import scipy.signal

from global_constants import *

def spectral_convolution(in1, in2):
    in1 = tf.cast(in1, tf.complex64)
    in2 = tf.cast(in2, tf.complex64)
    spec1 = tf.signal.fft(in1)
    spec2 = tf.signal.fft(in2)
    conv = tf.multiply(spec1, spec2)
    return tf.math.real(tf.signal.ifft(conv))

def inst_freq(inst_phase):
    return np.diff(inst_phase) / (2.0*np.pi) * len(inst_phase)

def inverse_hilbert_cos(amplitude_envelope, instantaneous_phase):
    T = len(instantaneous_phase)

    sp = np.fft.fft(instantaneous_phase)
    freq = np.fft.fftfreq(instantaneous_phase.shape[-1])
    cos = []
    for i in range(len(freq)):
        cos.append(amplitude_envelope[i] * np.sum((sp[-i]+sp[i]).real/(2*T) * np.cos(2.*np.pi*freq[i] + instantaneous_phase[i])))
    return cos
def inverse_hilbert_sin(amplitude_envelope, instantaneous_phase):
    T = len(instantaneous_phase)

    sp = np.fft.fft(instantaneous_phase)
    freq = np.fft.fftfreq(instantaneous_phase.shape[-1])
    sin = []
    for i in range(len(freq)):
        sin.append(amplitude_envelope[i] * np.sum((sp[-i]-sp[i]).imag/(2*T) * np.sin(2.*np.pi*freq[i] + instantaneous_phase[i])))
    return sin

def hilbert_from_scratch(u):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)

    N = len(u)
    # take forward Fourier transform
    U = np.fft.fft(u)
    M = N - N//2 - 1
    # zero out negative frequency components
    U[N//2+1:] = [0] * M
    # double fft energy except @ DC0
    U[1:N//2] = 2 * U[1:N//2]
    # take inverse Fourier transform
    v = np.fft.ifft(U)
    return v

def my_hilbert(inp):
    analytic_signal = tf.numpy_function(scipy.signal.hilbert, (tf.expand_dims(inp, axis=0)), tf.complex64)
    amplitude_envelope = tf.abs(analytic_signal)
    instantaneous_phase = tf.numpy_function(
        np.unwrap, tf.expand_dims(tf.numpy_function(np.angle,
        (tf.expand_dims(analytic_signal, axis=0)), tf.float64), axis=0), tf.float64)
    return tf.cast(amplitude_envelope, tf.float32), tf.cast(instantaneous_phase, tf.float32)

def inverse_hilbert(amplitude_envelope, instantaneous_phase):
    # close enough i guess
    return tf.concat(([0.], inst_freq(instantaneous_phase)), axis=0) * amplitude_envelope

def hilb_tensor(amp, phase):
    amp = tf.cast(amp, tf.float32)
    phase = tf.cast(phase, tf.float32)
    return tf.convert_to_tensor((amp, phase), dtype=tf.float32)
def invert_hilb_tensor(t):
    return inverse_hilbert(t[0], t[1])
