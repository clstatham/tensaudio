import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms
from helper import *
from global_constants import *
from hilbert import *
import os
import inspect

class TADiscriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(TADiscriminator, self).__init__(*args, **kwargs)
       
        self.loss = nn.BCELoss()

        self.n_layers = N_DIS_LAYERS
        self.ksz = DIS_KERNEL_SIZE
        self.ndf = DIS_N_MELS
        self.stride = 1

        self.net = []
        #self.net.append(nn.Identity())
        i = 0
        for _ in range(self.n_layers):
            c = 2**i
            n = 2**(i+1)
            
            self.net.append(nn.Conv2d(self.ndf*c, self.ndf*n, self.ksz, self.stride, 0, bias=False))
            self.net.append(nn.BatchNorm2d(self.ndf*n))
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
            
            i += 1
        n = 2**(i+1)
        self.net.append(nn.Flatten())
        #self.net.append(nn.Linear(8*self.ndf*n, 1))
        self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)
    
    def criterion(self, label, output):
        l = self.loss(output, label.cpu())
        #l = F.relu(l)
        return l

    def forward(self, inp):
        if DIS_MODE == 0:
            amp, phase = HilbertWithGradients.apply(inp)
            inp = torch.stack((amp, phase))
            inp = torch.unsqueeze(inp, 0)
            inp = torch.unsqueeze(inp, 3)
            out = self.net[0](inp.to(torch.float))
        elif DIS_MODE == 1:
            # what an ugly line of code
            mel1 = torchaudio.transforms.MelSpectrogram(SAMPLE_RATE, DIS_N_FFT, n_mels=DIS_N_MELS, normalized=True)(inp[:-DIS_N_MELS].to(torch.float).cpu()).cpu()
            #mel1.retain_grad()

        return self.net.forward(torch.unsqueeze(torch.unsqueeze(mel1,0),3))
