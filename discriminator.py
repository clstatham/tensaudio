import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torchaudio
import torchaudio.transforms
import pytorch_lightning as pl
#from online_norm_pytorch import OnlineNorm1d, OnlineNorm2d
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
            self.samps = TOTAL_SAMPLES_OUT * 2
            x = self.samps
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
        k1 = self.ksz
        s1 = self.stride
        k2 = self.ksz
        s2 = self.stride
        while self.samps > 64:
            c = min(DIS_MAX_CHANNELS, int(self.ndf * 2**i))
            n = min(DIS_MAX_CHANNELS, int(self.ndf * 2**(i+1)))
            
            # if y <= self.ksz:
            #     k2 = 1
            #     s2 = 1
            
            if x <= self.ksz:
                s1 = 1
                k1 = 1
            
            i += 1
            #sqrt_samps = int(np.sqrt(self.samps))
            x = (x - (k1 - 1) - 1) // s1 + 1
            #y = (y - (k2 - 1) - 1) // s2 + 1
            self.samps = x
            self.net.append(nn.Dropout(DIS_DROPOUT, False))
            self.net.append(nn.Conv1d(c, n, k1, s1, groups=c, bias=False))
            self.net.append(nn.BatchNorm1d(n))
            self.net.append(nn.LeakyReLU(0.2, inplace=False))
            #v_cprint("Created Conv2d layer #{6} with c={0} n={1} k=({2}, {3}) s=({4}, {5})\tsamps={7}".format(c, n, k1, s1, i, self.samps))
        print("Created", i, "sets of Discriminator layers.")
        x = (x - (k1 - 1) - 1) // s1 + 1
        #y = (y - (k2 - 1) - 1) // s2 + 1
        #self.samps = x*y
        self.net.append(nn.Conv1d(n, 1, 1, 1, groups=1, bias=False))
        self.net.append(nn.Flatten())
        self.net.append(nn.LazyLinear(32, bias=False))
        self.net.append(nn.LazyLinear(16, bias=False))
        self.net.append(nn.LazyLinear(8, bias=False))
        self.net.append(nn.LazyLinear(4, bias=False))
        self.net.append(nn.LazyLinear(2, bias=False))
        self.net.append(nn.LazyLinear(1, bias=False))
        self.net.append(nn.Sigmoid())
        #self.net.append(nn.Flatten())

        self.net = nn.ModuleList(self.net)
    
    def forward(self, inp):
        if DIS_MODE == 0:
            actual_input = inp.to(torch.float).view(inp.shape[0],2,-1)
        elif DIS_MODE == 1:
            raise NotImplementedError("NYI")
            # fft1 = ag.Variable(torch.unsqueeze(
            #     torch.fft.fft(inp.to(torch.float), norm='forward'), -1), requires_grad=True)
            # mag = ag.Variable(torch.square(torch.real(fft1)) + torch.square(torch.imag(fft1)), requires_grad=True)
            # pha = ag.Variable(torch.atan2(torch.real(fft1), torch.imag(fft1)), requires_grad=True)
            # actual_input = ag.Variable(torch.stack((mag, pha)).permute(0,1,2), requires_grad=True)
        elif DIS_MODE == 2:
            # what an ugly line of code
            if N_GEN_MEL_CHANNELS in inp.shape or DIS_N_MELS in inp.shape:
                actual_input = inp.to(torch.float).unsqueeze(0)
            else:
                melspec = AudioToMelWithGradients.apply(inp.to(torch.float), DIS_N_FFT, DIS_N_MELS, DIS_HOP_LEN).requires_grad_(True)
                if torch.is_grad_enabled():
                    melspec.retain_grad()
                actual_input = melspec.unsqueeze(1).requires_grad_(True)
                if torch.is_grad_enabled():
                    actual_input.retain_grad()
                # actual_input = ag.Variable(torch.unsqueeze(torch.unsqueeze(torchaudio.transforms.MelSpectrogram(
                #     SAMPLE_RATE, DIS_N_FFT, n_mels=DIS_N_MELS)(
                #             inp[:-DIS_N_MELS].to(torch.float).cpu()
                #         ),0),3), requires_grad=True)
            #mel1.retain_grad()
        elif DIS_MODE == 3:
            actual_input = audio_to_specgram(inp.view(inp.shape[0], -1)).view(inp.shape[0], self.ndf, -1).requires_grad_(True)

        verdicts = actual_input
        if torch.is_grad_enabled():
            verdicts.retain_grad()
        for i, layer in enumerate(self.net):
            if i >= len(self.net) - 4:
                #print("Checking")
                pass
            verdicts = layer(verdicts)
            # if layer.__class__.__name__.find('Conv') != -1:
            #     verdicts = F.normalize(verdicts)
            if torch.isnan(verdicts).any():
                verdicts = torch.nan_to_num(verdicts, 0.5, 0.5, 0.5)
        return verdicts.clamp(0.001+FAKE_LABEL, 0.999*REAL_LABEL+FAKE_LABEL).squeeze()
