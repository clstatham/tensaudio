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
                ksz=4, base_channels=N_CHANNELS, blk_channels=4, *args, **kwargs):
        super(DPAM_Discriminator, self).__init__(*args, **kwargs)
       
        self.loss = nn.L1Loss()

        self.ln_layers = []
        self.base_channels = base_channels
        self.blk_channels = blk_channels
        self.ksz = ksz
        for i in range(n_layers):
            self.ln_layers.append(nn.ReLU())
            self.ln_layers.append(nn.Conv1d(self.layer_channels_by_index(i), self.layer_channels_by_index(i+1), ksz, stride=1))
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
        loss_result=F.l1_loss(input=feat_current, target=feat_target)
        return loss_result

    def forward(self, input1_wav, clean1_wav):
        self.input1_wav = input1_wav
        self.clean1_wav = clean1_wav
        keep_prob=1.0
        
        # input1_wav = tf.expand_dims(input1_wav, axis=0)
        # input1_wav = tf.expand_dims(input1_wav, axis=0)
        # clean1_wav = tf.expand_dims(clean1_wav, axis=0)
        # clean1_wav = tf.expand_dims(clean1_wav, axis=0)
        channels = [self.layer_channels_by_index(i) for i in range(len(self.ln_layers))]
        input1_wav = prep_data_for_batch_operation(input1_wav, N_BATCHES, channels[0], None)
        clean1_wav = prep_data_for_batch_operation(clean1_wav, N_BATCHES, channels[0], None)

        others = self.featureloss_batch(input1_wav,clean1_wav)

        res=torch.sum(torch.tensor(others).cuda())
        res=torch.log(res)
        
        res=torch.sigmoid(res)
        dist_1=torch.reshape(res,[-1,1,1]).cuda()
        
        dist_1 = self.dense1(dist_1)
        dist_1 = self.dense2(dist_1)
        dist_1 = self.dense3(dist_1)
        dist_1 = self.dense4(dist_1)

        #self.net1 = tf.nn.softmax(self.dense1)
        net1 = torch.flatten(dist_1)
        return net1