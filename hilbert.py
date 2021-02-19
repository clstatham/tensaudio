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

class InstFreq(Function):
    @staticmethod
    def forward(ctx, p):
        pp = p.detach().cpu().numpy()
        result = np.diff(pp) / (2.0*np.pi) * len(pp) # hehe
        return p.new(result)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        numpy_go = grad_output.cpu().numpy()
        result = np.diff(numpy_go) / (2.0*np.pi) * len(numpy_go)
        return grad_output.new(result)

def inst_freq(inst_phase):
    return InstFreq.apply(inst_phase)

# def inverse_hilbert_cos(amplitude_envelope, instantaneous_phase):
#     T = len(instantaneous_phase)

#     sp = np.fft.fft(instantaneous_phase)
#     freq = np.fft.fftfreq(instantaneous_phase.shape[-1])
#     cos = []
#     for i in range(len(freq)):
#         cos.append(amplitude_envelope[i] * np.sum((sp[-i]+sp[i]).real/(2*T) * np.cos(2.*np.pi*freq[i] + instantaneous_phase[i])))
#     return cos
# def inverse_hilbert_sin(amplitude_envelope, instantaneous_phase):
#     T = len(instantaneous_phase)

#     sp = np.fft.fft(instantaneous_phase)
#     freq = np.fft.fftfreq(instantaneous_phase.shape[-1])
#     sin = []
#     for i in range(len(freq)):
#         sin.append(amplitude_envelope[i] * np.sum((sp[-i]-sp[i]).imag/(2*T) * np.sin(2.*np.pi*freq[i] + instantaneous_phase[i])))
#     return sin

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
    analytic_signal = scipy.signal.hilbert(inp.cpu())
    #analytic_signal = hilbert_from_scratch(inp)
    amplitude_envelope = torch.from_numpy(np.abs(analytic_signal)).cuda()
    instantaneous_phase = torch.from_numpy(np.unwrap(np.angle(analytic_signal))).cuda()
    return amplitude_envelope, instantaneous_phase

def inverse_hilbert(amplitude_envelope, instantaneous_phase):
    # close enough i guess
    p = []
    for t in instantaneous_phase:
        p.append(t)
    p = torch.stack(p)
    freq = inst_freq(p)
    #zero = torch.zeros_like(torch.tensor([1])).cuda()
    #freq = torch.cat((zero, freq), dim=0)
    return freq * torch.tensor(amplitude_envelope).cuda()

def invert_hilb_tensor(t):
    return inverse_hilbert(t[0], t[1])

def prep_hilb_for_denses(t):
    return torch.flatten(t)

def stack_hilb(t):
    return np.dstack((t.cpu()[0], t.cpu()[1]))

