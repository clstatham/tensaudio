import os
import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.transforms import InverseMelScale, MelSpectrogram, GriffinLim
#import nnAudio
#from nnAudio.Spectrogram import MelSpectrogram, Griffin_Lim

import torch.nn.functional as F
from torch.autograd import Function

import numba
import scipy
import scipy.optimize

from nnls import NNLSSolver, DTypeMatrix

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

def find_shape_from(actual, exp_batches, exp_channels, exp_timesteps):
    # order is important here for dumb reasons
    if exp_timesteps is None:
        exp_timesteps = np.max((1, (actual / (exp_channels * exp_batches))))
    if exp_batches is None:
        exp_batches = np.max((1, (actual / (exp_channels * exp_timesteps))))
    if exp_channels is None:
        exp_channels = np.max((1, (actual / (exp_batches * exp_timesteps))))
    target = int(exp_batches)*int(exp_channels)*int(exp_timesteps)
    if target != actual:
        # try again in case one was None
        t = np.max((1, (actual / (exp_channels * exp_batches))))
        b = np.max((1, (actual / (exp_channels * exp_timesteps))))
        c = np.max((1, (actual / (exp_batches * exp_timesteps))))
        target = int(t)*int(b)*int(c)
        if target == actual:
            return (int(t), int(b), int(c))
        raise Exception(str(target)+" != "+str(actual))
    return (int(exp_batches), int(exp_channels), int(exp_timesteps))

# def ensure_size(t, size, channels=1):
#     if channels == 1:
#         t = t.flatten()
#         delta = t.shape[0] - size
#         if delta < 0:
#         if type(t) == torch.Tensor:
#             t = torch.cat((t, torch.zeros(-delta)))
#         else:
#             t = np.concatenate((t, np.zeros(-delta)))
#         elif delta > 0:
#         t = t[:-delta]
#     else:
#         delta = t.shape[1]*channels - size
#         if delta < 0:
#             if type(t) == torch.Tensor:
#                 t = torch.tensor([torch.cat(t[i], torch.zeros(t[i].shape[0])) for i in range(channels)])
#             else:
#                 t = [np.concatenate(t[i], np.zeros(len(t[i]))) for i in range(channels)]
#         elif delta > 0:
#             t = t[:][:-delta]
#     return t, delta

class DataBatchPrep(Function):
    @staticmethod
    def forward(ctx, inp, exp_batches, exp_channels, exp_timesteps):
        ctx.set_materialize_grads(False)
        ctx.exp_batches = exp_batches
        ctx.exp_channels = exp_channels
        ctx.exp_timesteps = exp_timesteps
        #ctx.save_for_backward(torch.empty_like(inp))
        ctx.save_for_backward(inp)
        inp_ = inp.detach()
        return inp.new(prep_data_for_batch_operation(inp_, exp_batches, exp_channels, exp_timesteps, greedy=False, return_shape=False))
    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        orig = ctx.saved_tensors[0]
        return orig, None, None, None

def prep_data_for_batch_operation(t, exp_batches, exp_channels, exp_timesteps, greedy=False, return_shape=False):
    t = torch.flatten(t)
    actual = t.shape[0]
    b, c, s = None, None, None
    delta = 0
    try:
        b, c, s = find_shape_from(actual, exp_batches, exp_channels, exp_timesteps)
    except Exception as e:
        if greedy:
            # zero pad the result until it's valid
            while b is None or c is None or s is None:
                delta += 1
                try:
                    b, c, s = find_shape_from(actual+delta, exp_batches, exp_channels, exp_timesteps)
                except:
                    pass
            t = torch.cat((t, torch.zeros(delta)))
        else:
            raise e
    if greedy:
        if return_shape:
            return torch.reshape(torch.as_tensor(t), (b, c, s)), delta, (b,c,s)
        else:
            return torch.reshape(torch.as_tensor(t), (b, c, s)), delta
    else:
        if return_shape:
            return torch.reshape(torch.as_tensor(t), (b, c, s)), (b,c,s)
        else:
            return torch.reshape(torch.as_tensor(t), (b, c, s))
#return tf.reshape(t, (N_BATCHES, 2, 1))
def normalize_audio(audio):
    return F.normalize(audio.flatten(), dim=0)
def normalize_data(tensor):
    return tensor/(tensor.abs().max() + EPSILON)
def write_normalized_audio_to_disk(sig, fn):
    sig_numpy = librosa.util.normalize(sig.clone().detach().cpu().numpy())
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
    return  diff_pytorch(p) / (2.0*np.pi * p.size(time_dim))
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
    mag = torch.complex(mag, torch.zeros(1, dtype=mag.dtype).to(mag.device))
    phase = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))
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

def hilbert_from_scratch_pytorch(u):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)

    N = len(u.flatten())
    # take forward Fourier transform
    U = torch.fft.fft(u.flatten())
    M = N - N//2 - 1
    # zero out negative frequency components
    U[N//2+1:] = torch.zeros(M)
    # double fft energy except @ DC0
    U[1:N//2] = 2 * U[1:N//2]
    # take inverse Fourier transform
    v = torch.fft.ifft(U)
    if torch.isnan(v[0]):
        #raise RuntimeError
        pass
    return v

def stftgram_to_stft(stftgram):
    mag = stftgram[0]
    p = stftgram[1]
    phase_angle = torch.cumsum(p * np.pi, -1)
    #phase_angle = p * np.pi
    return polar_to_rect(mag, phase_angle)

def specgram_to_fft(specgram):
    mag = specgram[0]
    p = specgram[1]
    phase_angle = torch.cumsum(p * np.pi, -1)
    #phase_angle = p * np.pi
    return polar_to_rect(mag, phase_angle)

def audio_to_stftgram(audio, n_fft, hop_len):
    stft = torch.stft(audio, n_fft, hop_len, return_complex=True).requires_grad_(True)
    if torch.is_grad_enabled():
        stft.retain_grad()
    a = torch.abs(stft)
    p = torch.atan2(stft.imag, stft.real).requires_grad_(True)
    if torch.is_grad_enabled():
        a.retain_grad()
        p.retain_grad()
    f = inst_freq_pytorch_2(p, use_unwrap=False).requires_grad_(True)
    if torch.is_grad_enabled():
        f.retain_grad()
    return torch.stack((a, f), dim=0).requires_grad_(True)

def stftgram_to_audio(stftgram, hop_len):
    stft = stftgram_to_stft(stftgram).requires_grad_(True)
    if torch.is_grad_enabled():
        stft.retain_grad()
    out = torch.istft(stft, stft.shape[0], hop_len, return_complex=False).requires_grad_(True)
    if torch.is_grad_enabled():
        out.retain_grad()
    return out

def audio_to_specgram(audio):
    ana = hilbert_from_scratch_pytorch(audio)
    a = torch.abs(ana)
    p = torch.atan2(ana.imag, ana.real)
    f = inst_freq_pytorch_2(p, use_unwrap=False)
    return torch.stack((a, f), dim=0)

def specgram_to_audio(specgram):
    fft = specgram_to_fft(specgram).requires_grad_(True)
    if torch.is_grad_enabled():
        fft.retain_grad()
    invhilb = hilbert_from_scratch_pytorch(fft*-1).requires_grad_(True)
    if torch.is_grad_enabled():
        invhilb.retain_grad()
    out = torch.imag(invhilb).requires_grad_(True)
    if torch.is_grad_enabled():
        out.retain_grad()
    if torch.isnan(out[0]):
        #raise RuntimeError
        pass
    return out

class STFTToSpecgramWithGradients(Function):
    @staticmethod
    def forward(ctx, p):
        p_ = p.detach().clone().cpu().float()
        #stft = librosa.stft(p_, n_fft=n_fft)
        ctx.save_for_backward(p)
        specgram = stft_to_specgram(p_)
        #melspec = MelSpectrogram(SAMPLE_RATE, n_fft, hop_length=1, n_mels=n_mels)(p_)
        return p.new(specgram)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None
        #numpy_go = grad_output.cpu().numpy()
        return ctx.saved_tensors[0]

# class SpecgramToSTFTWithGradients(Function):
#     @staticmethod
#     def forward(ctx, p):
#         ctx.save_for_backward(p)
#         ctx.mark_dirty(p)
#         p = specgram_to_stft(p)
#         return p

#     @staticmethod
#     def backward(ctx, grad_output):
#         if grad_output is None:
#             return None
#         #numpy_go = grad_output.cpu().numpy()
#         return ctx.saved_tensors[0]


class AudioToStftWithGradients(Function):
    @staticmethod
    def forward(ctx, p, n_fft, hop_len):
        p_ = p.detach().clone().cpu().float().numpy()
        #stft = librosa.stft(p_, n_fft=n_fft)
        ctx.save_for_backward(p)
        stft = torch.from_numpy(librosa.stft(p_, n_fft=n_fft, hop_length=hop_len, center=True))
        #melspec = MelSpectrogram(SAMPLE_RATE, n_fft, hop_length=1, n_mels=n_mels)(p_)
        return p.new(stft)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None
        #numpy_go = grad_output.cpu().numpy()
        return ctx.saved_tensors[0], None, None

class AudioToMelWithGradients(Function):
    @staticmethod
    def forward(ctx, p, n_fft, n_mels, hop_len, power=2):
        p_ = p.detach().clone().squeeze().cpu().float().numpy()
        #stft = librosa.stft(p_, n_fft=n_fft)
        ctx.save_for_backward(p)
        stft = librosa.stft(p_, n_fft=n_fft, hop_length=hop_len)
        melspec = librosa.feature.melspectrogram(S=np.abs(stft), sr=SAMPLE_RATE, n_mels=n_mels, power=power, center=True)
        melspec = torch.from_numpy(melspec)
        #melspec = MelSpectrogram(SAMPLE_RATE, n_fft, hop_length=1, n_mels=n_mels)(p_)
        return p.new(melspec.to(p.device))

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        #numpy_go = grad_output.cpu().numpy()
        return ctx.saved_tensors[0], None, None, None

def stft_to_audio(stft, hop_len, n_iter):
    stft_ = stft.detach().clone().cpu().numpy()
    if n_iter > 0:
        print("Inverse STFT: Running Griffin-Lim for", n_iter, "iters...")
        audio = librosa.griffinlim(stft_, n_iter=n_iter, hop_length=hop_len, center=True)
    else:
        print("Inverse STFT: Taking ISTFT...")
        audio = librosa.istft(stft_, hop_length=hop_len, center=True)
    return torch.from_numpy(audio).to(audio.deive)

class MelToSTFTWithGradients(Function):
    @staticmethod
    def forward(ctx, p, n_mels=N_GEN_MEL_CHANNELS, n_fft=N_GEN_FFT, power=2):
        B = p.detach().clone().cpu().squeeze().numpy()
        ctx.save_for_backward(p)
        
        stft = torch.from_numpy(librosa.feature.inverse.mel_to_stft(B, sr=SAMPLE_RATE, n_fft=n_fft).astype(B.dtype))

        # with torch.enable_grad():
        #     A = librosa.filters.mel(SAMPLE_RATE, n_fft, n_mels=n_mels).astype(B.dtype)
        #     n_columns = 1

        #     x = np.linalg.lstsq(A, B, rcond=None)[0].astype(A.dtype)
        #     np.clip(x, 0, None, out=x)

        #     print("Inverse Mel: Calculating NNLS for", x.shape[-1], "iters...")
        #     for bl_s in range(0, x.shape[-1], n_columns):
        #         bl_t = min(bl_s + n_columns, B.shape[-1])
        #         bl_t = max(bl_t, 1)
        #         if bl_s == x.shape[-1]//4:
        #             print('Inverse Mel: 25% done...')
        #         elif bl_s == 2*x.shape[-1]//4:
        #             print('Inverse Mel: 50% done...')
        #         elif bl_s == 3*x.shape[-1]//4:
        #             print('Inverse Mel: 75% done...')
        #         input_b = B[:, bl_s].copy()
        #         result = scipy.optimize.nnls(A, input_b)[0]
        #         x[:, bl_s] = result
        #     stft = np.power(x, 1.0 / power)
        #     print("Inverse Mel: Done!")
        #with torch.enable_grad():
            #stft = torchaudio.transforms.InverseMelScale(n_fft, n_mels, SAMPLE_RATE)(B)
        #mel_filters = torchaudio.functional.create_fb_matrix(n_fft, 0, SAMPLE_RATE//2, n_mels, SAMPLE_RATE)
        #mel_filters = librosa.filters.mel(SAMPLE_RATE, n_fft, n_mels)
        #stft = torch.tensordot(B.transpose(-1,-2), mel_filters, 1)
        return p.new(stft.to(p.dtype))

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        #numpy_go = grad_output.cpu().numpy()
        return ctx.saved_tensors[0], None, None, None