import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchaudio
import torchaudio.transforms
import pytorch_lightning as pl
from online_norm_pytorch import OnlineNorm1d, OnlineNorm2d
from helper import *
from global_constants import *
from hilbert import *
import os
import inspect

class TADiscriminator(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(TADiscriminator, self).__init__(*args, **kwargs)

        self.ksz = DIS_KERNEL_SIZE
        self.stride = DIS_STRIDE
        if DIS_MODE in [0, 3]:
            self.ndf = 2
            x = int(np.sqrt(TOTAL_SAMPLES_OUT * 2))
            y = int(np.sqrt(TOTAL_SAMPLES_OUT * 2))
            self.samps = x*y
        elif DIS_MODE == 1:
            self.ndf = TOTAL_SAMPLES_OUT
        elif DIS_MODE == 2:
            self.ndf = 1
            x = DIS_N_MELS
            y = librosa.samples_to_frames(TOTAL_SAMPLES_OUT, DIS_HOP_LEN, DIS_N_FFT)
            self.samps = x*y
        else:
            raise ValueError("Invalid discriminator mode!")
        

        self.net = []
        #self.net.append(nn.Identity())
        i = 0
        c = self.ndf * 2
        n = self.ndf
        while self.samps > 1:
            c = min(DIS_MAX_CHANNELS, int(self.ndf * 2**i))
            n = min(DIS_MAX_CHANNELS, int(self.ndf * 2**(i+1)))
            i += 1
            sqrt_samps = int(np.sqrt(self.samps))
            x = (x - (1 - 1) - 1) // 2 + 1
            y = (y - (self.ksz - 1) - 1) // self.stride + 1
            self.samps = x*y
            self.net.append(nn.Conv2d(c, n, (1, self.ksz), (2, self.stride), groups=c, bias=False))
            self.net.append(OnlineNorm2d(n))
            self.net.append(nn.LeakyReLU(0.2, inplace=True))
        print("Created", i, "sets of Discriminator layers.")
        # sqrt_samps_d2 = int(np.sqrt(self.samps//2))
        # k = max(sqrt_samps_d2 // 2, 1)
        self.net.append(nn.Conv2d(n, 1, 1, 1, groups=1, bias=False))
        self.net.append(nn.Sigmoid())
        self.net.append(nn.Flatten())

        self.net = nn.ModuleList(self.net)
    
    def forward(self, inp):
        if DIS_MODE == 0:
            actual_input = ag.Variable(torch.unsqueeze(inp.to(torch.float), 0).reshape(1,2,TOTAL_SAMPLES_OUT//2), requires_grad=True)
        elif DIS_MODE == 1:
            fft1 = ag.Variable(torch.unsqueeze(
                torch.fft.fft(inp.to(torch.float), n=TOTAL_SAMPLES_OUT, norm='forward'), -1), requires_grad=True)
            mag = ag.Variable(torch.square(torch.real(fft1)) + torch.square(torch.imag(fft1)), requires_grad=True)
            pha = ag.Variable(torch.atan2(torch.real(fft1), torch.imag(fft1)), requires_grad=True)
            actual_input = ag.Variable(torch.stack((mag, pha)).permute(0,1,2), requires_grad=True)
        elif DIS_MODE == 2:
            # what an ugly line of code
            if N_GEN_MEL_CHANNELS in inp.shape or DIS_N_MELS in inp.shape:
                actual_input = inp.to(torch.float).unsqueeze(0)
            else:
                melspec = AudioToMelWithGradients.apply(inp.to(torch.float), DIS_N_FFT, DIS_N_MELS, DIS_HOP_LEN).requires_grad_(True)
                melspec.retain_grad()
                actual_input = melspec.unsqueeze(0).unsqueeze(0).requires_grad_(True)
                actual_input.retain_grad()
                # actual_input = ag.Variable(torch.unsqueeze(torch.unsqueeze(torchaudio.transforms.MelSpectrogram(
                #     SAMPLE_RATE, DIS_N_FFT, n_mels=DIS_N_MELS)(
                #             inp[:-DIS_N_MELS].to(torch.float).cpu()
                #         ),0),3), requires_grad=True)
            #mel1.retain_grad()
        elif DIS_MODE == 3:
            specgram = audio_to_specgram(inp.view(-1)).view(1, self.ndf, 2, -1)
            actual_input = ag.Variable(specgram, requires_grad=True)

        verdicts = actual_input
        verdicts.retain_grad()
        for i, layer in enumerate(self.net):
            # if layer.__class__.__name__.find('Conv') != -1:
                # if GEN_MODE == 6:
                #     verdicts = verdicts.view(1, layer.in_channels, 2, -1)
                # else:
                #     verdicts = verdicts.view(1, layer.in_channels, 2, -1)
            verdicts = layer(verdicts)
        return verdicts.clamp(0.001+FAKE_LABEL, 0.999*REAL_LABEL+FAKE_LABEL).squeeze().mean()
