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

from global_constants import *

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
#             t = torch.cat((t, torch.zeros(-delta).cuda()))
#         else:
#             t = np.concatenate((t, np.zeros(-delta)))
#         elif delta > 0:
#         t = t[:-delta]
#     else:
#         delta = t.shape[1]*channels - size
#         if delta < 0:
#             if type(t) == torch.Tensor:
#                 t = torch.tensor([torch.cat(t[i], torch.zeros(t[i].shape[0])) for i in range(channels)]).cuda()
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
        return inp.new(prep_data_for_batch_operation(inp_, exp_batches, exp_channels, exp_timesteps, greedy=False, return_shape=False).to(inp.device))
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
            t = torch.cat((t, torch.zeros(delta).cuda()))
        else:
            raise e
    if greedy:
        if return_shape:
            return torch.reshape(torch.as_tensor(t).cuda(), (b, c, s)), delta, (b,c,s)
        else:
            return torch.reshape(torch.as_tensor(t).cuda(), (b, c, s)), delta
    else:
        if return_shape:
            return torch.reshape(torch.as_tensor(t).cuda(), (b, c, s)), (b,c,s)
        else:
            return torch.reshape(torch.as_tensor(t).cuda(), (b, c, s))
#return tf.reshape(t, (N_BATCHES, 2, 1))
def normalize_audio(audio):
    return F.normalize(audio.flatten(), dim=0)
def normalize_data(tensor):
    return tensor/tensor.abs().max()
def write_normalized_audio_to_disk(sig, fn):
    sig_numpy = librosa.util.normalize(sig.clone().detach().cpu().numpy())
    scaled = np.int16(sig_numpy * 32767)
    soundfile.write(fn, scaled, SAMPLE_RATE, SUBTYPE)

class MelWithGradients(Function):
    @staticmethod
    def forward(ctx, p, n_fft, n_mels, hop_len):
        p_ = p.detach().clone().cpu().float().numpy()
        #stft = librosa.stft(p_, n_fft=n_fft)
        ctx.save_for_backward(p, torch.as_tensor(n_fft), torch.as_tensor(n_mels), torch.as_tensor(hop_len))
        melspec = librosa.feature.melspectrogram(p_, sr=SAMPLE_RATE, n_fft=n_fft, n_mels=n_mels, hop_length=hop_len)
        melspec = torch.from_numpy(melspec)
        #melspec = MelSpectrogram(SAMPLE_RATE, n_fft, hop_length=1, n_mels=n_mels)(p_)
        return p.new(melspec.to(p.device))

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        #numpy_go = grad_output.cpu().numpy()
        return ctx.saved_tensors[0], None, None, None

class InverseMelWithGradients(Function):
    @staticmethod
    def forward(ctx, p, n_fft, hop_len, n_iter):
        B = p.detach().clone().cpu().squeeze().numpy().astype(np.double)
        #stft = librosa.stft(p_, n_fft=n_fft)
        ctx.save_for_backward(p, torch.as_tensor(n_fft), torch.as_tensor(hop_len), torch.as_tensor(n_iter))
        #audio = librosa.feature.inverse.mel_to_audio(p_, sr=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_len)
        with torch.enable_grad():
            #stft = g_inverse_mel(p_.unsqueeze(0)).squeeze().numpy()
            #stft = torch.from_numpy(librosa.feature.inverse.mel_to_stft(A, sr=SAMPLE_RATE, n_fft=N_GEN_FFT))
            A = librosa.filters.mel(SAMPLE_RATE, n_fft, n_mels=N_GEN_MEL_CHANNELS).astype(np.double)

            MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10
            n_columns = MAX_MEM_BLOCK // (A.shape[0] * A.itemsize)
            n_columns = max(n_columns, 1)

            x = np.linalg.lstsq(A, B, rcond=None)[0].astype(A.dtype)
            np.clip(x, 0, None, out=x)
            x_init = x

            #print("A.shape:", A.shape)
            #print("B.shape:", B.shape)
            #print("x.shape:", x.shape)
            solver = NNLSSolver.instance()
            
            loss = torch.nn.L1Loss()
            for bl_s in range(0, x.shape[0], n_columns):
                if x.shape[1] - bl_s < 2:
                    bl_s = x.shape[0]
                    bl_t = x.shape[0]
                bl_t = min(bl_s + n_columns, x.shape[0])
                bl_t = max(bl_t, 1)
                print("Inverse Mel: Iter", bl_s, "/", x.shape[0])
                #input_x = x_init[:, bl_s:bl_t]
                #input_a = np.dot(A, input_x)
                input_b = B[:, bl_s:bl_t].copy()
                #print("input_a.shape:", A.shape)
                #print("input_b.shape:", input_b.shape)

                result = solver.solve(A, input_b, input_b.shape[1])
                #print("result.shape:", result.shape)
                x[:, bl_s:bl_t] = result

            stft = np.power(x, 1.0 / 2.0)
            print("Inverse Mel: Running Griffin-Lim for", n_iter, "iters...")
            audio = torch.from_numpy(librosa.griffinlim(stft, n_iter=n_iter, hop_length=1)).to(torch.float)
            print("Inverse Mel: Done!")
        return p.new(audio.to(p.device))

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        #numpy_go = grad_output.cpu().numpy()
        return ctx.saved_tensors[0], None, None, None