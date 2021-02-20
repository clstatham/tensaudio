import time
import ctcsound
import torch
from global_constants import *

# createChannel and ChannelUpdater credit to:
# Steven Yi <stevenyi@gmail.com>
# found at https://github.com/csound/csoundAPI_examples/blob/master/python/example10.py
# modifications by Connor Statham (github: clstatham)

def createChannel(cs, channelName):
    cs.channelPtr(channelName, 
        ctcsound.CSOUND_CONTROL_CHANNEL | ctcsound.CSOUND_INPUT_CHANNEL) 

class ChannelUpdater(object):
    def __init__(self, cs, channelName, updater):
        self.updater = updater
        self.cs = cs
        self.channelName = channelName
        createChannel(cs, channelName)

    def update(self):
        self.cs.setControlChannel(self.channelName, self.updater())

class CsoundInterface():
    def __init__(self):
        self.c = ctcsound.Csound()
        #self.c.SetOption("-odac")
        self.c.setOption("-b "+str(TOTAL_SAMPLES_OUT))
        self.total_instruments = 1
        self.total_gens = 1
        self.param1 = 0
        self.param2 = 0
        self.param3 = 0
        self.param4 = 0
        self.param5 = 0
        self.param6 = 0
        self.param7 = 0
        self.param8 = 0
        self.channels = [
            ChannelUpdater(self.c, "param1", self.get_param1),
            ChannelUpdater(self.c, "param2", self.get_param2),
            ChannelUpdater(self.c, "param3", self.get_param3),
            ChannelUpdater(self.c, "param4", self.get_param4),
            ChannelUpdater(self.c, "param5", self.get_param5),
            ChannelUpdater(self.c, "param6", self.get_param6),
            ChannelUpdater(self.c, "param7", self.get_param7),
            ChannelUpdater(self.c, "param8", self.get_param8),
        ]
        self.gens = """\n
        \n"""
        self.csinstruments = """\n
        sr="""+str(SAMPLE_RATE)+"""
        ksmps="""+str(KONTROL_SAMPLES)+"""
        nchnls="""+str(N_CHANNELS)+"""
        0dbfs=1

        gicos       ftgen   0, 0, 2^10, 11, 1

        instr 1
        kfr init 0
        khn init 0
        klh init 0
        kmul init 0
        iAAtt init 0
        iADec init 0
        iASus init 0
        iARel init 0
        kfr chnget "param1"
        knh chnget "param2"
        klh chnget "param3"
        kmul chnget "param4"
        iAAtt chnget "param5"
        iADec chnget "param6"
        iASus chnget "param7"
        iARel chnget "param8"
        aAmpEnv expsegr 0.0001,iAAtt,1,iADec,iASus,iARel,0.0001
        asig gbuzz 1, kfr, knh, klh, kmul, gicos
        outs asig*aAmpEnv
        endin
        \n"""
        self.csscore = """\n
        i1 0 """+str(OUTPUT_DURATION)+"""
        \n"""
    
    def set_param1(self, value):
        self.param1 = value
    def get_param1(self):
        return self.param1
    def set_param2(self, value):
        self.param2 = value
    def get_param2(self):
        return self.param2
    def set_param3(self, value):
        self.param3 = value
    def get_param3(self):
        return self.param3
    def set_param4(self, value):
        self.param4 = value
    def get_param4(self):
        return self.param4
    def set_param5(self, value):
        self.param5 = value
    def get_param5(self):
        return self.param5
    def set_param6(self, value):
        self.param6 = value
    def get_param6(self):
        return self.param6
    def set_param7(self, value):
        self.param7= value
    def get_param7(self):
        return self.param7
    def set_param8(self, value):
        self.param8 = value
    def get_param8(self):
        return self.param8

    def start(self):
        self.c.readScore(self.csscore)
        self.c.start()
    def stop(self):
        self.c.stop()
    def reset(self):
        self.c.reset()
    def compile(self):
        orc = self.csinstruments
        print(orc)
        return self.c.compileOrc(orc)
    def perform(self, params, window):
        for chn in self.channels:
            chn.update()
        done = False
        i = 1
        while not done:
            self.c.performKsmps()
            done = i*KONTROL_SAMPLES >= TOTAL_SAMPLES_OUT
            param = params[(i-1)*N_PARAMS:(i)*N_PARAMS]
            window.addstr(14,0, "Current Parameters:")
            window.move(15,0)
            window.clrtoeol()
            window.move(16,0)
            window.clrtoeol()
            window.addstr(15,0, str(round(param[0], 2))+"\t"+
                                str(round(param[1], 2))+"\t"+
                                str(round(param[2], 2))+"\t"+
                                str(round(param[3], 2))+"\t")
            window.addstr(16,0, str(round(param[4], 2))+"\t"+
                                str(round(param[5], 2))+"\t"+
                                str(round(param[6], 2))+"\t"+
                                str(round(param[7], 2))+"\t")
            self.update_params(param)
            for chn in self.channels:
                chn.update()
            #time.sleep(KONTROL_SAMPLES/SAMPLE_RATE)
            i += 1
        return self.c.outputBuffer()

    def update_params(self, params):
        params = params.flatten()
        if len(params) > N_PARAMS:
            raise ValueError("Incorrect number of parameters!")
        self.set_param1(params[0])
        self.set_param2(params[1])
        self.set_param3(params[2])
        self.set_param4(params[3])
        self.set_param5(params[4])
        self.set_param6(params[5])
        self.set_param7(params[6])
        self.set_param8(params[7])

    def create_gen(self, num):
        self.total_gens += 1
        out = """\n
        giSine      ftgen """+str(self.total_gens)+""" 0, 2^12, 10, 1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8
        gicos       ftgen   0, 0, 2^10, 11, 1
        \n"""
        self.gens += out
        return out

    def create_instrument(self):
        self.total_instruments += 1