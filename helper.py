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
def normalize_negone_one(x):
    a = x - x.min(-1, keepdim=True)[0]
    b = a / (a.max(-1, keepdim=True)[0] + EPSILON)
    return 2 * b - 1
def normalize_zero_one(x):
    a = x - x.min(-1, keepdim=True)[0]
    b = a / (a.max(-1, keepdim=True)[0] + EPSILON)
    return b
    
def write_normalized_audio_to_disk(sig, fn):
    sig_numpy = librosa.util.normalize(sig.numpy().astype(np.float32))
    #scaled = np.int16(sig_numpy * 32767)
    scipy.io.wavfile.write(fn, SAMPLE_RATE, sig_numpy)

def diff_pytorch(phi, dim=-1):
    return phi - F.pad(phi, (1, 0))[..., :dim]

def unwrap_pytorch(p, discont=np.pi, dim=-1):
    """Unwrap a cyclical phase tensor.
    (Ported to PyTorch from GANsynth project by clstatham)
    Args:
        p: Phase tensor.
        discont: Float, size of the cyclic discontinuity.
        axis: Axis of which to unwrap.
    Returns:
        unwrapped: Unwrapped tensor of same size as input.
    """
    dd = diff_pytorch(p, dim=dim)
    ddmod = (dd + np.pi) % (2.0 * np.pi) - np.pi
    idx = torch.logical_and(torch.eq(ddmod, -np.pi * torch.ones_like(ddmod)), torch.greater(dd, torch.zeros_like(dd)))
    ddmod = torch.where(idx, torch.ones_like(ddmod) * np.pi, ddmod)
    ph_correct = ddmod - dd
    idx = torch.less(torch.abs(dd), discont)
    ddmod = torch.where(idx, torch.zeros_like(ddmod), dd)
    ph_cumsum = torch.cumsum(ph_correct, dim=dim)

    shape = list(p.shape)
    shape[dim] = 1
    ph_cumsum = torch.cat([torch.zeros(shape, dtype=p.dtype), ph_cumsum], dim=dim)
    unwrapped = p + ph_cumsum[..., :p.shape[-1]]
    return unwrapped

def inst_freq(p):
    return np.diff(p) / (2.0*np.pi) * len(p)
def inst_freq_pytorch(p, time_dim=-1):
    dphase = diff_pytorch(p, dim=time_dim)
    dphase = torch.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
    dphase = torch.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)
    return dphase
def inst_freq_pytorch_2(phase_angle, time_dim=-1, use_unwrap=True):
    """Transform a fft tensor from phase angle to instantaneous frequency.
    Take the finite difference of the phase. Pad with initial phase to keep the
    tensor the same size.
    (Ported to PyTorch from GANsynth project by clstatham)
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
        phase_unwrapped = unwrap_pytorch(phase_angle, dim=time_dim)
        dphase = diff_pytorch(phase_unwrapped, dim=time_dim)
    else:
        # Keep dphase bounded. N.B. runs faster than a single mod-2pi expression.
        dphase = diff_pytorch(phase_angle, dim=time_dim)
        dphase = torch.where(dphase > np.pi, dphase - 2 * np.pi, dphase)
        dphase = torch.where(dphase < -np.pi, dphase + 2 * np.pi, dphase)

    # Add an initial phase to dphase.
    size = list(phase_angle.shape)
    size[time_dim] = 1
    begin = [0 for unused_s in size]
    phase_slice = torch.narrow(phase_angle, dim=0, start=0, length=size[0])
    dphase = torch.cat((phase_slice, dphase), dim=time_dim) / np.pi
    return dphase[1:]

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

def hilbert_from_scratch_pytorch(u):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)

    N = len(u.view(-1))
    # take forward Fourier transform
    U = torch.fft.fft(u.view(-1))
    M = N - N//2 - 1
    # zero out negative frequency components
    U[N//2+1:] = torch.zeros(M)
    # double fft energy except @ DC0
    U[1:N//2] = 2 * U[1:N//2]
    # take inverse Fourier transform
    v = torch.fft.ifft(U)
    return v

def stftgram_to_stft(stftgram):
    mag = stftgram[0].squeeze()
    p = stftgram[1].squeeze()
    #phase_angle = torch.cumsum(p * np.pi, -1)
    phase_angle = p * np.pi
    return polar_to_rect(mag, phase_angle)

def specgram_to_fft(specgram):
    mag = tf.squeeze(specgram[0])
    p = tf.squeeze(specgram[1])
    #phase_angle = torch.cumsum(p * np.pi, -1)
    phase_angle = mel_to_linear(p * np.pi)
    return polar_to_rect(mag, phase_angle)

def audio_to_stftgram(audio, n_fft, hop_len):
    out = []
    for batch in audio:
        real, imag = STFT(n_fft, hop_len).to(audio.device)(batch.unsqueeze(0))
        if torch.is_grad_enabled():
            real.requires_grad_(True)
            imag.requires_grad_(True)
            real.retain_grad()
            imag.retain_grad()
        a = torch.abs(polar_to_rect(real, imag))
        p = torch.atan2(imag, real).requires_grad_(True)
        if torch.is_grad_enabled():
            a.retain_grad()
            p.retain_grad()
        #f = inst_freq_pytorch(p).requires_grad_(True)
        p = p / np.pi
        #if torch.is_grad_enabled():
        #    f.retain_grad()
        out.append(torch.stack((a.squeeze().permute(1,0), p.squeeze().permute(1,0)), dim=0).requires_grad_(True).to(audio.device))
    return torch.stack(out).requires_grad_(True)

def stftgram_to_audio(stftgram, hop_len):
    out = []
    for batch in stftgram:
        stft = stftgram_to_stft(batch).requires_grad_(True).permute(1,0).unsqueeze(0).unsqueeze(0)
        if torch.is_grad_enabled():
            stft.retain_grad()
        out.append(ISTFT(2*(stft.shape[-1]-1), hop_len).to(stftgram.device)(stft.real, stft.imag, TOTAL_SAMPLES_OUT).to(stftgram.device).squeeze())
    return torch.stack(out).requires_grad_(True)

def audio_to_specgram(audio):
    out = []
    for batch in audio:
        ana = hilbert_from_scratch_pytorch(batch)
        mel_ana = ana
        a = torch.abs(mel_ana)
        p = linear_to_mel(torch.atan2(mel_ana.imag, mel_ana.real) / np.pi)
        #f = inst_freq_pytorch_2(p, use_unwrap=False)
        out.append(torch.stack((a, p), dim=0))
    return torch.stack(out)

def specgram_to_audio(specgram):
    out = []
    for batch in specgram:
        fft = specgram_to_fft(batch)
        invhilb = np.fft.ifft(fft)
        out.append(np.real(invhilb))
    return tf.stack(out)
