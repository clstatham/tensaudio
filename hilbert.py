import six
import numpy as np
import torch
import torchaudio
import scipy.signal

from global_constants import *

def spectral_convolution(in1, in2):
    spec1 = np.fft.fft(in1)
    spec2 = np.fft.fft(in2)
    conv = np.multiply(spec1, spec2)
    return np.real(np.fft.ifft(conv))

def inst_freq(inst_phase):
    t = inst_phase.clone().detach().cpu()
    return np.diff(t) / (2.0*np.pi) * len(t)

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
    analytic_signal = scipy.signal.hilbert(inp)
    #analytic_signal = hilbert_from_scratch(inp)
    amplitude_envelope = torch.from_numpy(np.abs(analytic_signal)).cuda()
    instantaneous_phase = torch.from_numpy(np.unwrap(np.angle(analytic_signal))).cuda()
    return amplitude_envelope, instantaneous_phase

def inverse_hilbert(amplitude_envelope, instantaneous_phase):
    # close enough i guess
    return torch.cat((torch.tensor([0.]).cuda(), torch.from_numpy(inst_freq(instantaneous_phase)).cuda()), dim=0) * amplitude_envelope

def invert_hilb_tensor(t):
    return inverse_hilbert(t[0], t[1])

def prep_hilb_for_denses(t):
    return torch.flatten(t)

def prep_hilb_for_batch_operation(t, exp_batches, exp_timesteps, exp_units):
    # this took me FUCKING FOREVER TO FIGURE OUT
    tmp = torch.flatten(t).shape[0]
    total = exp_batches*exp_timesteps*exp_units
    assert(total == tmp)
    batches_ratio = exp_batches / total
    timesteps_ratio = exp_timesteps / total
    units_ratio = exp_units / total
    return torch.reshape(torch.as_tensor(t).cuda(), (int(tmp*batches_ratio), int(tmp*timesteps_ratio), int(tmp*units_ratio)))

def stack_hilb(t):
    return np.dstack((t.cpu()[0], t.cpu()[1]))

