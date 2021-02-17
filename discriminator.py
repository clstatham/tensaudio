from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from network_model import *
from helper import *
from global_constants import *
from hilbert import inverse_hilbert
import os
import inspect

class DPAM_Discriminator(tf.keras.Model):
    def __init__(self, type='scratch', *args, **kwargs):
        super(DPAM_Discriminator, self).__init__(*args, **kwargs)
        self.type = type

        self.loss = tf.keras.losses.mean_absolute_error

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=DISCRIMINATOR_LR)
       
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self)
        self.manager = tf.train.CheckpointManager(self.ckpt, os.path.join(MODEL_DIR, "dis_ckpts"), max_to_keep=1)
        
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint)
            print("Restored DPAM from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing DPAM from scratch.")
    
    def call(self, input1_wav, clean1_wav):
        ## Training Parameters
        SE_LAYERS = N_DIS_LAYERS # NUMBER OF INTERNAL LAYERS
        SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
        SE_LOSS_LAYERS = N_DIS_LAYERS # NUMBER OF FEATURE LOSS LAYERS
        
        # FEATURE LOSS NETWORK
        LOSS_LAYERS = N_DIS_LAYERS # NUMBER OF INTERNAL LAYERS
        LOSS_BASE_CHANNELS = 32 # NUMBER OF FEATURE CHANNELS PER LAYER IN FIRT LAYER
        LOSS_BLK_CHANNELS = 5 # NUMBER OF LAYERS BETWEEN CHANNEL NUMBER UPDATES
        LOSS_NORM =  'SBN' # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

        FILTER_SIZE=3

        self.input1_wav = input1_wav
        self.clean1_wav = clean1_wav
        keep_prob=1.0
        
        input1_wav = tf.expand_dims(input1_wav, axis=0)
        input1_wav = tf.expand_dims(input1_wav, axis=0)
        clean1_wav = tf.expand_dims(clean1_wav, axis=0)
        clean1_wav = tf.expand_dims(clean1_wav, axis=0)
        input1_wav = prep_audio_for_batch_operation(input1_wav)
        clean1_wav = prep_audio_for_batch_operation(clean1_wav)

        others,loss_sum = featureloss_batch(input1_wav,clean1_wav,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE) 

        res=tf.reduce_sum(others,0)
        distance=K.log(res)
        
        distance=tf.nn.sigmoid(distance)
        dist_1=tf.reshape(distance,[-1,1,1])
        
        self.dense1=tf.keras.layers.Dense(16)(dist_1)
        self.dense2=tf.keras.layers.Dense(6)(self.dense1)
        self.dense3=tf.keras.layers.Dense(2)(self.dense2)
        self.dense4=tf.keras.layers.Dense(1)(self.dense3)
        #self.net1 = tf.nn.softmax(self.dense1)
        self.net1 = K.flatten(self.dense4)
        return self.net1