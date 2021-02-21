import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import *
from global_constants import *
from hilbert import *
import os
import inspect

class DPAM_Discriminator(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DPAM_Discriminator, self).__init__(*args, **kwargs)
       
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

        # self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        # self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(MODEL_DIR, "dis_ckpts"), max_to_keep=1)
        
        # if self.manager.latest_checkpoint:
        #     self.ckpt.restore(self.manager.latest_checkpoint)
        #     cprint("Restored DPAM from {}".format(self.manager.latest_checkpoint))
        # else:
        #     cprint("Initializing DPAM from scratch.")
    
    def criterion(self, label, output):
        l = self.loss(torch.unsqueeze(output, 0), torch.unsqueeze(label, 0))
        l = F.relu(l)
        return l

    def layer_channels_by_index(self, i):
        return np.max((1, self.base_channels * (i)))

    def call_ln(self, inp):
        out = inp.float()
        i = 1
        for layer in self.ln_layers:
            n_channels = self.layer_channels_by_index(i)
            out = layer(out)
            i += 1
        return out

    def featureloss_batch(self, target, current):
        feat_current = self.call_ln(current)
        feat_target = self.call_ln(target)
        c_sum = torch.sum(torch.abs(feat_current))
        t_sum = torch.sum(torch.abs(feat_target))
        loss_result=F.l1_loss(input=c_sum, target=t_sum)
        return loss_result

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
            real, imag = STFTWithGradients.apply(input1).detach().cuda()
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