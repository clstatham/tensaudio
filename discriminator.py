import numpy as np
import torch
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
        self.ndf = 2

        self.net = []
        i = 0
        for _ in range(self.n_layers):
            c = 2**i
            n = 2**(i+1)
            self.net.append(nn.Conv2d(self.ndf*c, self.ndf*n, self.ksz, 2, 1, bias=False))
            self.net.append(nn.BatchNorm2d(self.ndf*n))
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
            i += 1

        self.net.append(nn.Conv2d(self.ndf*(2**i), 1, self.ksz, 1, 0, bias=False))
        self.net.append(nn.Sigmoid())

        self.net = nn.ModuleList(self.net)
    
    def criterion(self, label, output):
        l = self.loss(torch.unsqueeze(output, 0), torch.unsqueeze(label, 0))
        l = F.relu(l)
        return l

    def forward(self, input1):
        if type(input1) != torch.Tensor:
            input1 = torch.from_numpy(input1).float().cuda()
        if DIS_MODE == 0:
            amp, phase = HilbertWithGradients.apply(input1).detach().cuda()
            hilb = torch.stack((amp, phase))
            hilb = torch.unsqueeze(hilb, 0)
            hilb = torch.unsqueeze(hilb, 3)
            out = self.net[0](hilb)
        elif DIS_MODE == 1:
            real, imag = FFTWithGradients.apply(input1).detach().cuda()
            stft1 = torch.stack((real, imag))
            stft1 = torch.unsqueeze(stft1, 0)
            stft1 = torch.unsqueeze(stft1, 3)
            out = self.net[0](stft1)
        
        i = 1
        for layer in self.net[1:]:
            #cprint("In discriminator layer #", i)
            out = layer(out)
            i += 1
        #out = torch.mean(out)
        return out