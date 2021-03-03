import six
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd import Function
from global_constants import *

def spectral_convolution(in1, in2):
    spec1 = np.fft.fft(in1.cpu())
    spec2 = np.fft.fft(in2.cpu())
    conv = np.multiply(spec1, spec2)
    return np.real(np.fft.ifft(conv))

def inverse_hilbert_cos(amplitude_envelope, instantaneous_phase):
    T = len(instantaneous_phase)

    sp = torch.fft.fft(instantaneous_phase)
    freq = np.fft.fftfreq(instantaneous_phase.detach().clone().cpu().numpy().shape[-1])
    cos = []
    for i in range(len(freq)):
        cos.append(amplitude_envelope[i] * torch.sum(torch.real(sp[-i]+sp[i])/(2*T) * torch.cos(2.*np.pi*freq[i] + instantaneous_phase[i])))
    return torch.as_tensor(cos)
def inverse_hilbert_sin(amplitude_envelope, instantaneous_phase):
    T = len(instantaneous_phase)

    sp = torch.fft.fft(instantaneous_phase)
    freq = np.fft.fftfreq(instantaneous_phase.detach().clone().cpu().numpy().shape[-1])
    sin = []
    for i in range(len(freq)):
        sin.append(amplitude_envelope[i] * torch.sum(torch.imag(sp[-i]-sp[i])/(2*T) * torch.sin(2.*np.pi*freq[i] + instantaneous_phase[i])))
    return torch.as_tensor(sin)
def inverse_hilbert_pytorch(amplitude_envelope, instantaneous_phase):
    # cos = inverse_hilbert_cos(amplitude_envelope, instantaneous_phase)
    # sin = inverse_hilbert_sin(amplitude_envelope, instantaneous_phase)
    # assert(len(cos) == len(sin))
    # out = []
    # for i in range(len(cos)):
    #    out.append(cos[i] + sin[i])
    # return torch.as_tensor(out)
    return torch.cat((torch.zeros(1).cuda(), inst_freq_pytorch(instantaneous_phase))) * amplitude_envelope

class HilbertWithGradients(Function):
    @staticmethod
    def forward(ctx, p):
        p_ = p.detach().cpu().numpy()
        #stft = librosa.stft(p_, n_fft=n_fft)
        ctx.save_for_backward(p)
        hilb = np.imag(scipy.signal.hilbert(p_))
        return p.new(hilb)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
          return None
        #numpy_go = grad_output.cpu().numpy()
        return ctx.saved_tensors[0]

# class InverseHilbertWithGradients(Function):
#     @staticmethod
#     def forward(ctx, a, p):
#         a_ = f.detach().cpu().numpy()
#         p_ = p.detach().cpu().numpy()
#         #stft = librosa.stft(p_, n_fft=n_fft)
#         ctx.save_for_backward(a)
#         ctx.save_for_backward(p)
#         hilb = inverse_hilbert(a, p)
#         return p.new(hilb)

#     @staticmethod
#     def backward(ctx, grad_output):
#         if grad_output is None:
#           return None
#         #numpy_go = grad_output.cpu().numpy()
#         return ctx.saved_tensors[0]

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

def hilbert_from_scratch_pytorch(u, return_amp_phase=True):
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
    if return_amp_phase:
        amp_env = torch.abs(torch.imag(v))
        phi = torch.angle(torch.imag(v))

        # credit to Past-Future-AI @ Pytorch forums for this one
        dphi = diff_pytorch(phi)
        dphi_m = ((dphi+np.pi) % (2*np.pi)) - np.pi
        dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
        phi_adj = dphi_m-dphi
        phi_adj[dphi.abs()<np.pi] = 0
        inst_phas = phi + phi_adj.cumsum(-1)

        return amp_env.cuda(), inst_phas.cuda()
    else:
        return v.cuda()

def my_hilbert(inp):
    analytic_signal = scipy.signal.hilbert(inp.detach().cpu().numpy())
    #analytic_signal = hilbert_from_scratch(inp)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return amplitude_envelope, instantaneous_phase

# def inverse_hilbert(amplitude_envelope, instantaneous_phase):
#     # close enough i guess
#     freq = inst_freq(instantaneous_phase)
#     return freq * amplitude_envelope

# def inverse_hilbert_pytorch(amplitude_envelope, instantaneous_phase):
#     # close enough i guess
#     #freq = inst_freq_pytorch(instantaneous_phase.flatten())
#     # v = freq * amplitude_envelope
#     return inverse_hilbert(freq, instantaneous_phase)

def invert_hilb_tensor(t):
    return inverse_hilbert(t[0], t[1])

def prep_hilb_for_denses(t):
    return torch.flatten(t)

def stack_hilb(t):
    return np.dstack((t.cpu()[0], t.cpu()[1]))
