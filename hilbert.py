import six
import numpy as np
import scipy.signal
import torch
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
