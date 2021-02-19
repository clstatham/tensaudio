from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import *
from global_constants import *
from hilbert import inverse_hilbert
import os
import inspect

class DPAM_Discriminator(nn.Module):
    def __init__(self, n_layers=N_DIS_LAYERS, reuse=False, norm_type="SBN",
                ksz=DIS_KERNEL_SIZE, base_channels=N_DIS_CHANNELS, blk_channels=4, *args, **kwargs):
        super(DPAM_Discriminator, self).__init__(*args, **kwargs)
       
        self.loss = nn.BCELoss()

        self.ln_layers = []
        self.base_channels = base_channels
        self.blk_channels = blk_channels
        self.ksz = ksz
        #self.ndf = 1 + N_FFT // 2
        self.ndf = 2
        # for i in range(n_layers):
        #     self.ln_layers.append(nn.ReLU())
        #     self.ln_layers.append(nn.Conv2d(self.layer_channels_by_index(i), self.layer_channels_by_index(i+1), ksz, stride=1))
        #     #self.ln_layers.append(nn.Dropout(1.0))
        # self.ln_layers = nn.ModuleList(self.ln_layers)

        self.net = []
        self.net.append(nn.Conv2d(self.ndf, self.ndf*2, self.ksz, 2, 1, bias=False))
        self.net.append(nn.BatchNorm2d(self.ndf*2))
        self.net.append(nn.LeakyReLU(0.2, inplace=True))

        self.net.append(nn.Conv2d(self.ndf*2, self.ndf*4, self.ksz, 2, 1, bias=False))
        self.net.append(nn.BatchNorm2d(self.ndf*4))
        self.net.append(nn.LeakyReLU(0.2, inplace=True))

        self.net.append(nn.Conv2d(self.ndf*4, self.ndf*8, self.ksz, 2, 1, bias=False))
        self.net.append(nn.BatchNorm2d(self.ndf*8))
        self.net.append(nn.LeakyReLU(0.2, inplace=True))

        self.net.append(nn.Conv2d(self.ndf*8, 1, self.ksz, 1, 0, bias=False))
        self.net.append(nn.Sigmoid())
        self.net = nn.ModuleList(self.net)

        # self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        # self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(MODEL_DIR, "dis_ckpts"), max_to_keep=1)
        
        # if self.manager.latest_checkpoint:
        #     self.ckpt.restore(self.manager.latest_checkpoint)
        #     print("Restored DPAM from {}".format(self.manager.latest_checkpoint))
        # else:
        #     print("Initializing DPAM from scratch.")
    
    def criterion(self, ex, op):
        return self.loss(torch.unsqueeze(torch.mean(op), 0), torch.sigmoid(torch.unsqueeze(torch.mean(ex), 0)))

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
        real, imag = STFTWithGradients.apply(input1, N_FFT).cuda()
        stft1 = torch.stack((real, imag))
        stft1 = torch.unsqueeze(stft1, 0)
        #input1 = prep_data_for_batch_operation(input1, None, input1.shape[0], 1).float()
        #print("In discriminator layer #", 0)
        out = self.net[0](stft1)
        i = 1
        for layer in self.net[1:]:
            #print("In discriminator layer #", i)
            out = layer(out)
            i += 1
        return out