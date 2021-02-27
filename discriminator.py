import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
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
        if DIS_MODE in [1, 3]:
            self.ndf = 2
        elif DIS_MODE == 2:
            self.ndf = DIS_N_MELS
        else:
            self.ndf = 1
        self.stride = DIS_STRIDE

        self.net = []
        #self.net.append(nn.Identity())
        i = 0
        for _ in range(self.n_layers):
            c = 2**i
            n = 2**(i+1)
            i += 1
            if DIS_MODE in [0, 1, 3]:
                self.net.append(nn.Conv1d(self.ndf*c, self.ndf*n, self.ksz, self.stride, 0, bias=False).cuda())
                if i < self.n_layers:
                    self.net.append(nn.BatchNorm1d(self.ndf*n).cuda())
            else:
                self.net.append(nn.Conv2d(self.ndf*c, self.ndf*n, self.ksz, self.stride, 0, bias=False).cuda())
                if i < self.n_layers:
                    self.net.append(nn.BatchNorm2d(self.ndf*n).cuda())
            if i < self.n_layers:
                self.net.append(nn.LeakyReLU(0.2, inplace=True).cuda())
            
        n = 2**i
        #self.net.append(nn.Flatten().cuda())
        if DIS_MODE in [0, 1, 3]:
            self.net.append(nn.Conv1d(self.ndf*n, 1, 1, self.stride, 0, bias=False).cuda())
        else:
            self.net.append(nn.Conv2d(self.ndf*n, 1, 1, self.stride, 0, bias=False).cuda())
        #self.net.append(nn.Softsign().cuda())
        self.net.append(nn.Sigmoid().cuda())
        self.net.append(nn.Flatten().cuda())
        
        

        self.net = nn.Sequential(*self.net)
    
    def criterion(self, label, output):
        return self.loss(output.cuda(), label.cuda())

    def forward(self, inp):
        if DIS_MODE == 0:
            actual_input = ag.Variable(torch.unsqueeze(inp.clone().to(torch.float), 0).reshape(1,1,TOTAL_SAMPLES_OUT), requires_grad=True).cuda()
        elif DIS_MODE == 1:
            fft1 = ag.Variable(torch.unsqueeze(
                torch.fft.fft(inp.to(torch.float), n=TOTAL_SAMPLES_OUT, norm='forward').cuda(), -1).cuda(), requires_grad=True).cuda()
            mag = ag.Variable(torch.square(torch.real(fft1)) + torch.square(torch.imag(fft1)), requires_grad=True).cuda()
            pha = ag.Variable(torch.atan2(torch.real(fft1), torch.imag(fft1)), requires_grad=True).cuda()
            actual_input = ag.Variable(torch.stack((mag, pha)).permute(2,0,1), requires_grad=True).cuda()
        elif DIS_MODE == 2:
            # what an ugly line of code
            actual_input = ag.Variable(torch.unsqueeze(torch.unsqueeze(torchaudio.transforms.MelSpectrogram(
                SAMPLE_RATE, DIS_N_FFT, n_mels=DIS_N_MELS)(
                        inp[:-DIS_N_MELS].to(torch.float).cpu()
                    ).cuda(),0),3).cuda(), requires_grad=True).cuda()
            #mel1.retain_grad()
        elif DIS_MODE == 3:
            amp_env, inst_phas = hilbert_from_scratch_pytorch(inp.to(torch.float), n=TOTAL_SAMPLES_OUT)
            actual_input = ag.Variable(torch.unsqueeze(torch.stack((amp_env, inst_phas)), -1).cuda().permute(2,0,1), requires_grad=True).cuda()

        verdicts = self.net.forward(actual_input)
        return verdicts.clamp(0.001+FAKE_LABEL, 0.999*REAL_LABEL+FAKE_LABEL).squeeze()
