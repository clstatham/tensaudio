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
        if DIS_MODE == 0:
            self.ndf = 2
        elif DIS_MODE in [1, 3]:
            self.ndf = TOTAL_SAMPLES_OUT
        elif DIS_MODE == 2:
            self.ndf = DIS_N_MELS
        else:
            raise ValueError("Invalid discriminator mode!")
        self.stride = DIS_STRIDE

        self.net = []
        #self.net.append(nn.Identity())
        i = 0
        c = self.ndf * 2
        n = self.ndf
        for _ in range(self.n_layers):
            if DIS_MODE in [0, 1, 3]:
                c = int(self.ndf * 2**i)
                n = int(self.ndf * 2**(i-4))
                i -= 4
                self.net.append(nn.Conv1d(c, n, self.ksz, self.stride, 0, bias=False).to(0))
                if i < self.n_layers:
                    self.net.append(nn.BatchNorm1d(n).to(0))
            else:
                c = int(self.ndf * 2**i)
                n = int(self.ndf * 2**(i-1))
                i -= 1
                self.net.append(nn.Conv1d(c, n, self.ksz, 1, 0, bias=False).to(0))
                if i < self.n_layers:
                    self.net.append(nn.BatchNorm1d(n).to(0))
            if i < self.n_layers:
                self.net.append(nn.LeakyReLU(0.2, inplace=True).to(0))
            
        #self.net.append(nn.Flatten().to(0))
        if DIS_MODE in [0, 1, 3]:
            self.net.append(nn.Conv1d(n, 1, 1, 1, 0, bias=False).to(0))
        else:
            self.net.append(nn.Conv1d(n, 1, 1, 1, 0, bias=False).to(0))
        #self.net.append(nn.Softsign().to(0))
        self.net.append(nn.Sigmoid().to(0))
        self.net.append(nn.Flatten().to(0))
        
        

        self.net = nn.Sequential(*self.net)
    
    def criterion(self, label, output):
        return self.loss(output.to(0), label.to(0))

    def forward(self, inp):
        if DIS_MODE == 0:
            actual_input = ag.Variable(torch.unsqueeze(inp.clone().to(torch.float), 0).reshape(1,2,TOTAL_SAMPLES_OUT//2), requires_grad=True).to(0)
        elif DIS_MODE == 1:
            fft1 = ag.Variable(torch.unsqueeze(
                torch.fft.fft(inp.to(torch.float), n=TOTAL_SAMPLES_OUT, norm='forward').to(0), -1).to(0), requires_grad=True).to(0)
            mag = ag.Variable(torch.square(torch.real(fft1)) + torch.square(torch.imag(fft1)), requires_grad=True).to(0)
            pha = ag.Variable(torch.atan2(torch.real(fft1), torch.imag(fft1)), requires_grad=True).to(0)
            actual_input = ag.Variable(torch.stack((mag, pha)).permute(0,1,2), requires_grad=True).to(0)
        elif DIS_MODE == 2:
            # what an ugly line of code
            if N_GEN_MEL_CHANNELS in inp.shape or DIS_N_MELS in inp.shape:
                actual_input = inp.clone().to(torch.float).to(0).unsqueeze(0)
            else:
                melspec = AudioToMelWithGradients.apply(inp.clone().to(torch.float), DIS_N_FFT, DIS_N_MELS, DIS_HOP_LEN)
                actual_input = melspec.unsqueeze(0).to(0)
                # actual_input = ag.Variable(torch.unsqueeze(torch.unsqueeze(torchaudio.transforms.MelSpectrogram(
                #     SAMPLE_RATE, DIS_N_FFT, n_mels=DIS_N_MELS)(
                #             inp[:-DIS_N_MELS].to(torch.float).cpu()
                #         ).to(0),0),3).to(0), requires_grad=True).to(0)
            #mel1.retain_grad()
        elif DIS_MODE == 3:
            amp_env, inst_phas = hilbert_from_scratch_pytorch(inp.to(torch.float), n=TOTAL_SAMPLES_OUT)
            actual_input = ag.Variable(torch.unsqueeze(torch.stack((amp_env, inst_phas)), -1).to(0).permute(2,0,1), requires_grad=True).to(0)

        verdicts = self.net.forward(actual_input)
        return verdicts.clamp(0.001+FAKE_LABEL, 0.999*REAL_LABEL+FAKE_LABEL).squeeze()
