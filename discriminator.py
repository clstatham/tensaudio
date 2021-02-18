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
                ksz=2, base_channels=32, blk_channels=5, *args, **kwargs):
        super(DPAM_Discriminator, self).__init__(*args, **kwargs)
       
        self.loss = nn.L1Loss()

        self.ln_layers = []
        self.base_channels = base_channels
        self.blk_channels = blk_channels
        n_channels_last = 0
        for i in range(n_layers):
            n_channels = base_channels * (2 ** (i // blk_channels)) # UPDATE CHANNEL COUNT
            
            self.ln_layers.append(nn.ReLU())
            if i == 0:
                self.ln_layers.append(nn.Conv1d(N_TIMESTEPS, n_channels, ksz, stride=1))
            else:
                self.ln_layers.append(nn.Conv1d(n_channels_last, n_channels, ksz, stride=1))
            
            n_channels_last = n_channels
            #self.ln_layers.append(nn.Dropout(1.0))
        self.ln_layers = nn.ModuleList(self.ln_layers)

        self.dense1=nn.Linear(1, 16)
        self.dense2=nn.Linear(16, 6)
        self.dense3=nn.Linear(6, 2)
        self.dense4=nn.Linear(2, 1)

        # self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        # self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(MODEL_DIR, "dis_ckpts"), max_to_keep=1)
        
        # if self.manager.latest_checkpoint:
        #     self.ckpt.restore(self.manager.latest_checkpoint)
        #     print("Restored DPAM from {}".format(self.manager.latest_checkpoint))
        # else:
        #     print("Initializing DPAM from scratch.")
    
    def criterion(self, ex, op):
        op = torch.flatten(op)
        return self.loss(ex, op)

    def call_ln(self, inp):
        out = inp
        for layer in self.ln_layers:
            out = layer(out)
        return out

    def featureloss_batch(self, target, current):
        feat_current = self.call_ln(current)

        feat_target = self.call_ln(target)
        
        loss_vec = []
        
        channels = np.asarray([self.base_channels * (2 ** (i // self.blk_channels)) for i in range(len(self.ln_layers))])
        
        for i in range(len(self.ln_layers)):
            loss_result=l2_loss(feat_target[i], feat_current[i])
            loss_vec.append(loss_result)
        
        return loss_vec

    def forward(self, input1_wav, clean1_wav):
        self.input1_wav = input1_wav
        self.clean1_wav = clean1_wav
        keep_prob=1.0
        
        # input1_wav = tf.expand_dims(input1_wav, axis=0)
        # input1_wav = tf.expand_dims(input1_wav, axis=0)
        # clean1_wav = tf.expand_dims(clean1_wav, axis=0)
        # clean1_wav = tf.expand_dims(clean1_wav, axis=0)
        input1_wav = prep_audio_for_batch_operation(input1_wav, N_BATCHES, N_TIMESTEPS, N_UNITS)
        clean1_wav = prep_audio_for_batch_operation(clean1_wav, N_BATCHES, N_TIMESTEPS, N_UNITS)

        others = self.featureloss_batch(input1_wav,clean1_wav)

        res=torch.sum(torch.tensor(others))
        distance=torch.log(res)
        
        distance=torch.sigmoid(distance)
        dist_1=torch.reshape(distance,[-1,1,1]).cuda()
        
        dist_1 = self.dense1(dist_1)
        dist_1 = self.dense2(dist_1)
        dist_1 = self.dense3(dist_1)
        dist_1 = self.dense4(dist_1)

        #self.net1 = tf.nn.softmax(self.dense1)
        net1 = torch.flatten(dist_1)
        return net1