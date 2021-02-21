import time
import ctcsound
import torch
import librosa
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
    def __init__(self, vis):
        self.params_sec = OUTPUT_DURATION/TOTAL_PARAM_UPDATES
        self.secs_param = TOTAL_PARAM_UPDATES/OUTPUT_DURATION
        self.c = ctcsound.Csound()
        self.vis = vis
        #self.t = ctcsound.CsoundPerformanceThread(self.c.csound())
        #self.c.createMessageBuffer(True)
        #self.c.SetOption("-odac")
        self.c.setOption("-b "+str(TOTAL_SAMPLES_OUT))
        self.total_instruments = 1
        self.total_gens = 1
        self.params = [1.] * N_PARAMS
        self.channels = [
            ChannelUpdater(self.c, "param"+str(i), lambda: self.get_param(i)) for i in range(1, N_PARAMS+1)
        ]

        #for i in range(N_PARAMS):
        
        
        self.gens = """\n
        \n"""
        self.csinstruments = """\n
        sr="""+str(SAMPLE_RATE)+"""
        ksmps="""+str(KONTROL_SAMPLES)+"""
        nchnls="""+str(1)+"""
        0dbfs=1
        
        """+FM_SYNTH
        
        
    
    def set_param(self, num, value):
        self.params[num-1] = value
    def get_param(self, num):
        return self.params[num-1]

    def param_gen(self, num):
        while True:
            if num in [1]:
                yield np.log2(self.get_param(num))*5
            else:
                yield self.get_param(num)*10

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

        #self.vis.reset()
        i = 1
        st = [0,0]
        tim = ctcsound.RtClock()
        self.c.initTimerStruct(tim)
        while not done:
            # self.c.reset()
            # self.compile()
            param_slice = params[(i-1)*N_PARAMS:(i)*N_PARAMS]
            if len(param_slice) == 0:
                i = 1
                param_slice = params[0:N_PARAMS]
            for param in param_slice:
                if param < 0.1:
                    i += 1
                    continue
            param_slice = self.scale_params(param_slice)
            self.update_params(param_slice)
            if window is not None:
                try:
                    window.addstr(14,0, "Current Parameters:")
                    window.move(15,0)
                    window.clrtoeol()
                    window.move(16,0)
                    window.clrtoeol()
                    window.move(17,0)
                    window.clrtoeol()
                    window.move(18,0)
                    window.clrtoeol()
                    window.move(19,0)
                    window.clrtoeol()
                    window.addstr(15,0, str(round(self.params[0]   ))+"\t"+
                                        str(round(self.params[1], 2))+"\t"+
                                        str(round(self.params[2], 2))+"\t"+
                                        str(round(self.params[3], 2))+"\t"+
                                        str(round(self.params[4], 2))+"\t"+
                                        str(round(self.params[5], 2))+"\t"+
                                        str(round(self.params[6], 2))+"\t"+
                                        str(round(self.params[7], 2))+"\t"+
                                        str(round(self.params[8], 2))+"\t"+
                                        str(round(self.params[9], 2))+"\t")
                    window.addstr(16,0, str(round(self.params[10], 2))+"\t"+
                                        str(round(self.params[11], 2))+"\t"+
                                        str(round(self.params[12], 2))+"\t"+
                                        str(round(self.params[13], 2))+"\t"+
                                        str(round(self.params[14], 2))+"\t"+
                                        str(round(self.params[15], 2))+"\t"+
                                        str(round(self.params[16], 2))+"\t"+
                                        str(round(self.params[17], 2))+"\t"+
                                        str(round(self.params[18], 2))+"\t"+
                                        str(round(self.params[19], 2))+"\t")
                    window.addstr(17,0, str(round(self.params[20], 2))+"\t"+
                                        str(round(self.params[21], 2))+"\t"+
                                        str(round(self.params[22], 2))+"\t"+
                                        str(round(self.params[23], 2))+"\t"+
                                        str(round(self.params[24], 2))+"\t"+
                                        str(round(self.params[25], 2))+"\t"+
                                        str(round(self.params[26], 2))+"\t"+
                                        str(round(self.params[27], 2))+"\t"+
                                        str(round(self.params[28], 2))+"\t"+
                                        str(round(self.params[29], 2))+"\t")
                    window.addstr(18,0, str(round(self.params[30], 2))+"\t"+
                                        str(round(self.params[31], 2))+"\t"+
                                        str(round(self.params[32], 2))+"\t"+
                                        str(round(self.params[33], 2))+"\t"+
                                        str(round(self.params[34], 2))+"\t"+
                                        str(round(self.params[35], 2))+"\t"+
                                        str(round(self.params[36], 2))+"\t"+
                                        str(round(self.params[37], 2))+"\t"+
                                        str(round(self.params[38], 2))+"\t"+
                                        str(round(self.params[39], 2))+"\t")
                    window.addstr(19,0, str(round(self.params[40], 2))+"\t"+
                                        str(round(self.params[41], 2))+"\t"+
                                        str(round(self.params[42], 2))+"\t"+
                                        str(round(self.params[43], 2))+"\t"+
                                        str(round(self.params[44], 2))+"\t"+
                                        str(round(self.params[45], 2))+"\t"+
                                        str(round(self.params[46], 2))+"\t"+
                                        str(round(self.params[47], 2))+"\t"+
                                        str(round(self.params[48], 2))+"\t"+
                                        str(round(self.params[49], 2))+"\t")

                    
                    
                    window.refresh()
                    time.sleep(0.001)
                except curses.error:
                    pass

            for chn in self.channels:
                chn.update()
            score = self.update_score()
            print(score)
            self.c.readScore(score)
            self.c.rewindScore()
            self.c.start()
            res = self.c.perform()
            #done = res != 0
            cur_tim = self.c.CPUTime(tim)
            out = np.concatenate((out, self.c.outputBuffer()))
            done = len(out) >= TOTAL_SAMPLES_OUT
            #self.set_param(17, self.get_param(1))
            #self.set_param(18, self.get_param(2))
            #time.sleep(self.params_sec)
            i += 1
        self.c.stop()
        self.c.cleanup()
        return out

    def scale_params(self, params):
        params = params.flatten()
        if len(params) != N_PARAMS:
            raise ValueError("Incorrect number of parameters!")
        adsr_fac = self.params_sec * 4
        ratio_fac = params[50] * 8.
        params[0] *= 440. * 4.
        params[0] += 110.
        #params[0] *= 60                # fr
        #params[0] += 12
        #params[0] = 440. * np.power(2, (round(params[0])-49) / 12.)
        params[1:30] *= adsr_fac
        params[30:40] *= 1.             # amp
        params[40:50] *= ratio_fac
        #params[40:50] = np.round(params[40:50])
        return params

    def update_params(self, params):
        params = params.flatten()
        if len(params) != N_PARAMS:
            raise ValueError("Incorrect number of parameters!")
        for i in range(N_PARAMS-2):
            self.set_param(i+1, params[i])
        self.vis.update_vals()
        self.vis.update_display()
    
    def update_score(self):
        out = """\n
        i1 0 """+str(OUTPUT_DURATION)
        for param in self.params:
            out += " "+str(param)
        out += "\n"
        return out

    def create_gen(self, num):
        self.total_gens += 1
        out = """"""
        self.gens += out
        return out

    def create_instrument(self):
        self.total_instruments += 1

FM_SYNTH = """
gisine	ftgen 0, 0, 2^12, 10, 1	;A SINE WAVE

instr 	1
	kporttime	linseg	0,0.01,0.05		;PORTAMENTO TIME RAMPS UP QUICKLY FROM ZERO

	icps	=	p4
	kfund		portk	icps, kporttime		;SMOOTH VARIABLE CHANGES WITH PORTK
	kamp	=	0.1

	kampatt1	= p5
	kampatt2	= p6
	kampatt3	= p5
	kampatt4	= p7
	kampatt5	= p8
	kampatt6	= p9
	kampatt7	= p10
	kampatt8	= p11
	kampatt9	= p12
	kampatt10	= p13
	kampdec1	= p14
	kampdec2	= p15
	kampdec3	= p16
	kampdec4	= p17
	kampdec5	= p18
	kampdec6	= p19
	kampdec7	= p20
	kampdec8	= p21
	kampdec9	= p22
	kampdec10	= p23
	kamprel1	= p24
	kamprel2	= p25
	kamprel3	= p26
	kamprel4	= p27
	kamprel5	= p28
	kamprel6	= p29
	kamprel7	= p30
	kamprel8	= p31
	kamprel9	= p32
	kamprel10	= p33

	kPartAmp1	= p34
	kPartAmp2	= p35
	kPartAmp3	= p36
	kPartAmp4	= p37
	kPartAmp5	= p38
	kPartAmp6	= p39
	kPartAmp7	= p40
	kPartAmp8	= p41
	kPartAmp9	= p42
	kPartAmp10	= p43

	kratio1		= p44
	kratio2		= p45
	kratio3		= p46
	kratio4		= p47
	kratio5		= p48
	kratio6		= p49
	kratio7		= p50
	kratio8		= p51
	kratio9		= p52
	kratio10	= p53
	
	

	;AMPLITUDE ENVELOPES (WITH MIDI RELEASE SEGMENT)
	aenv1		expsegr	.001, i(kampatt1), 1.001, i(kampdec1), .001, i(kamprel1), .001
	aenv2		expsegr	.001, i(kampatt2), 1.001, i(kampdec2), .001, i(kamprel2), .001
	aenv3		expsegr	.001, i(kampatt3), 1.001, i(kampdec3), .001, i(kamprel3), .001
	aenv4		expsegr	.001, i(kampatt4), 1.001, i(kampdec4), .001, i(kamprel4), .001
	aenv5		expsegr	.001, i(kampatt5), 1.001, i(kampdec5), .001, i(kamprel5), .001
	aenv6		expsegr	.001, i(kampatt6), 1.001, i(kampdec6), .001, i(kamprel6), .001
	aenv7		expsegr	.001, i(kampatt7), 1.001, i(kampdec7), .001, i(kamprel7), .001
	aenv8		expsegr	.001, i(kampatt8), 1.001, i(kampdec8), .001, i(kamprel8), .001
	aenv9		expsegr	.001, i(kampatt9), 1.001, i(kampdec9), .001, i(kamprel9), .001
	aenv10		expsegr	.001, i(kampatt10),1.001, i(kampdec10),.001, i(kamprel10), .001
	
	;SEPARATE OSCILLATORS CREATE EACH OF THE PARTIALS (NOTE THAT FLTK VERTICAL SLIDERS ARE INVERTED TO ALLOW MINIMUM VALUES TO BE LOWEST ON THE SCREEN)
	;OUTPUT		OPCODE	AMPLITUDE	                        | FREQUENCY    | FUNCTION_TABLE
	apart1		foscili	kamp*(aenv1  - 0.001) * (1-kPartAmp1),  kfund,kratio1, 	1,gisine
	apart2		foscili	kamp*(aenv2  - 0.001) * (1-kPartAmp2),  kfund,kratio2, 	1,gisine
	apart3		foscili	kamp*(aenv3  - 0.001) * (1-kPartAmp3),  kfund,kratio3, 	1,gisine
	apart4		foscili	kamp*(aenv4  - 0.001) * (1-kPartAmp4),  kfund,kratio4, 	1,gisine
	apart5		foscili	kamp*(aenv5  - 0.001) * (1-kPartAmp5),  kfund,kratio5, 	1,gisine
	apart6		foscili	kamp*(aenv6  - 0.001) * (1-kPartAmp6),  kfund,kratio6, 	1,gisine
	apart7		foscili	kamp*(aenv7  - 0.001) * (1-kPartAmp7),  kfund,kratio7, 	1,gisine
	apart8		foscili	kamp*(aenv8  - 0.001) * (1-kPartAmp8),  kfund,kratio8, 	1,gisine
	apart9		foscili	kamp*(aenv9  - 0.001) * (1-kPartAmp9),  kfund,kratio9, 	1,gisine
	apart10     foscili	kamp*(aenv10 - 0.001) * (1-kPartAmp10), kfund,kratio10, 1,gisine
	                                                          
	;SUM THE 10 OSCILLATORS:
	amix		sum	apart1,\
				apart2,\
				apart3,\
				apart4,\
				apart5,\
				apart6,\
				apart7,\
				apart8,\
				apart9,\
				apart10
				
		out		amix	;SEND MIXED SIGNAL TO THE OUTPUTS
endin		
"""