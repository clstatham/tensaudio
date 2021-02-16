import six
import numpy as np
import tensorflow as tf
import scipy.signal

from global_constants import *

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
    analytic_signal = scipy.signal.hilbert(inp)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return amplitude_envelope, instantaneous_phase

def _stft(inp):
    if type(inp) is np.ndarray:
        inp_numpy = inp
    else:
        inp_numpy = np.array(inp)
    r = librosa.stft(inp_numpy, n_fft=N_FFT)
    return r

def stft(inp):
    return tf.numpy_function(_stft, [inp], tf.complex64)

def _istft(inp):
    if type(inp) is np.ndarray:
        inp_numpy = inp
    else:
        inp_numpy = np.array(inp)
    return librosa.istft(inp_numpy)
def istft(inp):
    return tf.numpy_function(_istft, [inp], tf.float32)

def spectral_convolution(in1, in2):
    in1 = tf.cast(in1, tf.complex64)
    in2 = tf.cast(in2, tf.complex64)
    spec1 = tf.signal.fft(in1)
    spec2 = tf.signal.fft(in2)
    conv = tf.multiply(spec1, spec2)
    return tf.math.real(tf.signal.ifft(conv))

def inst_freq(inst_phase):
    return (np.diff(inst_phase) / (2.0*np.pi) * len(inst_phase))

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
def inverse_hilbert(amplitude_envelope, instantaneous_phase):
    # cos = inverse_hilbert_cos(amplitude_envelope, instantaneous_phase)
    # sin = inverse_hilbert_sin(amplitude_envelope, instantaneous_phase)
    # assert(len(cos) == len(sin))
    #out = []
    #for i in range(len(cos)):
    #    out.append(cos[i] + sin[i])
    #return out
    return np.concatenate(([0], inst_freq(instantaneous_phase))) * amplitude_envelope

def hilb_tensor(amp, phase):
    amp = tf.cast(amp, tf.float32)
    phase = tf.cast(phase, tf.float32)
    return tf.convert_to_tensor((amp, phase), dtype=tf.float32)
def invert_hilb_tensor(t):
    return inverse_hilbert(t[0], t[1])
