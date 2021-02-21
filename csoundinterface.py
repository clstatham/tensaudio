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
        self.params_sec = OUTPUT_DURATION/TOTAL_PARAM_UPDATES
        self.secs_param = TOTAL_PARAM_UPDATES/OUTPUT_DURATION
        self.c = ctcsound.Csound()
        #self.c.createMessageBuffer(True)
        #self.c.SetOption("-odac")
        self.c.setOption("-b "+str(TOTAL_SAMPLES_OUT))
        self.total_instruments = 1
        self.total_gens = 1
        self.params = [0.] * N_PARAMS
        self.channels = [
            ChannelUpdater(self.c, "param"+str(i), lambda: self.get_param(i)) for i in range(1, N_PARAMS+1)
        ]
        
        self.gens = """\n
        \n"""
        self.csinstruments = """\n
        sr="""+str(SAMPLE_RATE)+"""
        ksmps="""+str(KONTROL_SAMPLES)+"""
        ;nchnls="""+str(N_CHANNELS)+"""
        nchnls=1
        0dbfs=1

        gicos       ftgen   0, 0, 2^12, 11, 1

        instr 1
        iFr init 0
        iNh init 0
        iCut init 0
        iRes init 0
        iAAtt init 0
        iADec init 0
        iASus init 0
        iARel init 0
        iFAtt init 0
        iFDec init 0
        iFSus init 0
        iFRel init 0
        iRAtt init 0
        iRDec init 0
        iRSus init 0
        iRRel init 0
        iFr = p4
        iNh = p5
        iCut = p6
        iRes = p7
        iAAtt = p8
        iADec = p9
        iASus = p10
        iARel = p11
        iFAtt = p12
        iFDec = p13
        iFSus = p14
        iFRel = p15
        iRAtt = p16
        iRDec = p17
        iRSus = p18
        iRRel = p19
        

        kFilEnv adsr iFAtt,iFDec,iFSus,iFRel
        kAmpEnv adsr iAAtt,iADec,iASus,iARel
        kFreqEnv adsr iRAtt,iRDec,iRSus,iRRel
        asig buzz 1, iFr, kFreqEnv*iNh, gicos
        ;asig vco2 1, iFr, 2
        afil moogladder asig, kFilEnv*iCut, iRes
        outs afil*kAmpEnv
        endin
        \n"""
    
    def set_param(self, num, value):
        self.params[num-1] = value
    def get_param(self, num):
        return self.params[num-1]

    def start(self):
        self.c.start()
    def stop(self):
        self.c.stop()
    def reset(self):
        self.c.reset()
    def compile(self):
        orc = self.csinstruments
        #print(orc)
        return self.c.compileOrc(orc)

    def perform(self, params, window):
        for chn in self.channels:
            chn.update()
        out = np.array([])
        done = False
        last_params = params[0:N_PARAMS]
        self.update_params(last_params)
        self.c.reset()
        self.compile()

        for chn in self.channels:
            chn.update()
        
        i = 0
        while not done:
            self.c.reset()
            self.compile()
            param_slice = params[(i-1)*N_PARAMS:(i)*N_PARAMS]
            if len(param_slice) == 0:
                param_slice = last_params
            else:
                param_slice = self.scale_params(param_slice)
            self.update_params(param_slice)

            try:
                window.addstr(14,0, "Current Parameters:")
                window.move(15,0)
                window.clrtoeol()
                window.move(16,0)
                window.clrtoeol()
                window.addstr(15,0, str(round(self.params[0]   ))+"\t"+
                                    str(round(self.params[1]   ))+"\t"+
                                    str(round(self.params[2]   ))+"\t"+
                                    str(round(self.params[3], 2))+"\t")
                window.addstr(16,0, str(round(self.params[4], 2))+"\t"+
                                    str(round(self.params[5], 2))+"\t"+
                                    str(round(self.params[6], 2))+"\t"+
                                    str(round(self.params[7], 2))+"\t")
                window.addstr(15,40,str(round(self.params[8], 2))+"\t"+
                                    str(round(self.params[9], 2))+"\t"+
                                    str(round(self.params[10], 2))+"\t"+
                                    str(round(self.params[11], 2))+"\t")
                window.addstr(16,40,str(round(self.params[12], 2))+"\t"+
                                    str(round(self.params[13], 2))+"\t"+
                                    str(round(self.params[14], 2))+"\t"+
                                    str(round(self.params[15], 2))+"\t")
                window.refresh()
                time.sleep(0.001)
            except curses.error:
                pass

            for chn in self.channels:
                chn.update()
            score = self.update_score()
            print(score)
            self.c.readScore(score)
            self.c.start()

            self.c.perform()
            done = i*self.secs_param >= OUTPUT_DURATION
            out = np.concatenate((out, self.c.outputBuffer()))
            i += 1
            #time.sleep(0.01)
        #if done < 0:
        #    raise Exception("Error in performance!")
        self.c.stop()
        
        #done = i*self.params_sec >= OUTPUT_DURATION
        #param_slice[4:8] = initial_params[4:8] # ignore changes to the ADSR
        #time.sleep(KONTROL_SAMPLES/SAMPLE_RATE)
        #print(len(out))
        return out

    def scale_params(self, params):
        params = params.flatten()
        if len(params) != N_PARAMS:
            raise ValueError("Incorrect number of parameters!")
        adsr_fac = self.secs_param
        params[0] *= SAMPLE_RATE / 4    # fr
        params[0] += 1
        params[1] *= 64                 # nh
        params[1] += 1
        params[1] = int(params[1])
        params[2] *= SAMPLE_RATE        # Cut
        params[3] *= 1.                 # Res
        params[4:16] *= adsr_fac        # Envelopes
        params[4:16] += 0.01
        return params

    def update_params(self, params):
        params = params.flatten()
        if len(params) != N_PARAMS:
            raise ValueError("Incorrect number of parameters!")
        for i in range(N_PARAMS):
            self.set_param(i+1, params[i])
    
    def update_score(self):
        """
        iFr = p4
        iNh = p5
        iCut = p6
        iRes = p7
        iAAtt = p8
        iADec = p9
        iASus = p10
        iARel = p11
        iFAtt = p12
        iFDec = p13
        iFSus = p14
        iFRel = p15
        iRAtt = p16
        iRDec = p17
        iRSus = p18
        iRRel = p19
        """
        return ("""\n
        i1 0 """+str(OUTPUT_DURATION)+
            " "+str(self.params[0])+
            " "+str(self.params[1])+
            " "+str(self.params[2])+
            " "+str(self.params[3])+
            " "+str(self.params[4])+
            " "+str(self.params[5])+
            " "+str(self.params[6])+
            " "+str(self.params[7])+
            " "+str(self.params[8])+
            " "+str(self.params[9])+
            " "+str(self.params[10])+
            " "+str(self.params[11])+
            " "+str(self.params[12])+
            " "+str(self.params[13])+
            " "+str(self.params[14])+
            " "+str(self.params[15])+"""
        \n""")

    def create_gen(self, num):
        self.total_gens += 1
        out = """"""
        self.gens += out
        return out

    def create_instrument(self):
        self.total_instruments += 1