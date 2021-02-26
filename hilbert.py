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

def inst_freq(p):
    return np.diff(p) / (2.0*np.pi) * len(p)

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

def hilbert_from_scratch_pytorch(u, n=TOTAL_SAMPLES_OUT):
    # N : fft length
    # M : number of elements to zero out
    # U : DFT of u
    # v : IDFT of H(U)

    N = len(u.flatten())
    # take forward Fourier transform
    U = torch.fft.fft(u.flatten(), n=n)
    M = N - N//2 - 1
    # zero out negative frequency components
    U[N//2+1:] = torch.zeros(M)
    # double fft energy except @ DC0
    U[1:N//2] = 2 * U[1:N//2]
    # take inverse Fourier transform
    v = torch.fft.ifft(U, n=n)
    amp_env = torch.abs(v)
    phi = torch.angle(v)

    # credit to Past-Future-AI @ Pytorch forums for this one
    dphi = F.pad(phi[..., 1:]-phi[..., :-1], (1,0))
    dphi_m = ((dphi+np.pi) % (2*np.pi)) - np.pi
    dphi_m[(dphi_m==-np.pi)&(dphi>0)] = np.pi
    phi_adj = dphi_m-dphi
    phi_adj[dphi.abs()<np.pi] = 0
    inst_phas = phi + phi_adj.cumsum(-1)

    return amp_env.cuda(), inst_phas.cuda()

def my_hilbert(inp):
    analytic_signal = scipy.signal.hilbert(inp.detach().cpu().numpy())
    #analytic_signal = hilbert_from_scratch(inp)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    return amplitude_envelope, instantaneous_phase

def inverse_hilbert(amplitude_envelope, instantaneous_phase):
    # close enough i guess
    freq = inst_freq(instantaneous_phase)
    #zero = torch.zeros_like(torch.tensor([1])).cuda()
    #freq = torch.cat((zero, freq), dim=0)
    return freq * amplitude_envelope

def invert_hilb_tensor(t):
    return inverse_hilbert(t[0], t[1])

def prep_hilb_for_denses(t):
    return torch.flatten(t)

def stack_hilb(t):
    return np.dstack((t.cpu()[0], t.cpu()[1]))
