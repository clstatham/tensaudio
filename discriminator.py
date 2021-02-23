import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
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
        self.ndf = 1

        self.net = []
        #self.net.append(nn.Identity())
        i = 0
        for _ in range(self.n_layers//2):
            c = 2**i
            n = 2**(i+1)
            self.net.append(nn.Conv2d(self.ndf*c, self.ndf*n, self.ksz, 2, 1, bias=False))
            self.net.append(nn.BatchNorm2d(self.ndf*n))
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
            i += 1

        self.net.append(nn.Conv2d(self.ndf*(2**i), 1, self.ksz, 1, 0, bias=False))
        self.net.append(nn.Flatten())

        linear_units = int(((2*TOTAL_SAMPLES_OUT)+((self.n_layers/16)*KONTROL_SAMPLES))/(2**i))
        self.net2 = []
        for _ in range(self.n_layers//2):
            self.net2.append(nn.Linear(linear_units, linear_units))
            self.net2.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.net2.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)
        self.net2 = nn.Sequential(*self.net2)
    
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
            #mag, phas = FFTWithGradients.apply(input1)
            fft1 = torch.fft.fft(inp)
            real, imag = torch.real(fft1), torch.imag(fft1)
            real.retain_grad()
            imag.retain_grad()
            mag = torch.sqrt(real**2 + imag**2)
            pha = torch.atan2(real, imag)
            mag.retain_grad()
            pha.retain_grad()
            stacked = torch.stack([mag, pha]).cpu()
            stacked.retain_grad()

        out1 = self.net.forward(torch.unsqueeze(torch.unsqueeze(stacked, 0), 0).to(torch.float))
        out2 = self.net2.forward(out1)
        return out2
