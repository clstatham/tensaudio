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

    def get(self):
        result = self.cs.controlChannel(self.channelName)
        if not result[1]:
            return result[0]
    
    def update(self):
        self.cs.setControlChannel(self.channelName, next(self.updater))

class CsoundInterface():
    def __init__(self, vis):
        self.params_sec = OUTPUT_DURATION/TOTAL_PARAM_UPDATES
        self.secs_param = TOTAL_PARAM_UPDATES/OUTPUT_DURATION
        self.c = ctcsound.Csound()
        self.vis = vis
        #self.t = ctcsound.CsoundPerformanceThread(self.c.csound())
        #self.c.createMessageBuffer(True)
        #self.c.setOption("-n")
        #self.c.setOption("-+rtmidi=mme")
        self.c.setOption("-+rtaudio=mme")
        self.c.setOption("-odac1")
        self.c.setOption("-M0")
        #self.c.setOption("-b "+str(512))
        self.total_instruments = 1
        self.total_gens = 1
        self.params = [1.] * N_PARAMS
        self.channels = [
            ChannelUpdater(self.c, "param"+str(i), self.yield_param(i)) for i in range(1, N_PARAMS+1)
        ]

        self.gens = """\n
        \n"""
        self.csinstruments = """
        sr="""+str(SAMPLE_RATE)+"""
        ksmps="""+str(KONTROL_SAMPLES)+"""
        nchnls="""+str(1)+"""
        0dbfs=1
        \n"""+FM_SYNTH
        
        
    
    def set_param(self, num, value):
        self.params[num-1] = value
    def get_param(self, num):
        return self.params[num-1]
    def yield_param(self, num):
        while True:
            yield self.params[num-1]

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

    def update_channels(self):
        for chn in self.channels:
            chn.update()

    def perform(self, params, window):
        
        out = np.array([])
        done = False
        last_params = params[0:N_PARAMS]
        self.update_params(last_params)
        self.update_channels()
        self.c.reset()
        self.compile()
        self.c.start()

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
            self.update_channels()
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
                except curses.error:
                    pass
                try:
                    window.addstr(15,0, str(round(self.channels[0].get()   ))+"\t"+
                                        str(round(self.channels[1].get(), 2))+"\t"+
                                        str(round(self.channels[2].get(), 2))+"\t"+
                                        str(round(self.channels[3].get(), 2))+"\t"+
                                        str(round(self.channels[4].get(), 2))+"\t"+
                                        str(round(self.channels[5].get(), 2))+"\t"+
                                        str(round(self.channels[6].get(), 2))+"\t"+
                                        str(round(self.channels[7].get(), 2))+"\t"+
                                        str(round(self.channels[8].get(), 2))+"\t"+
                                        str(round(self.channels[9].get(), 2))+"\t")
                except curses.error:
                    pass
                try:
                    window.addstr(16,0, str(round(self.channels[10].get(), 2))+"\t"+
                                        str(round(self.channels[11].get(), 2))+"\t"+
                                        str(round(self.channels[12].get(), 2))+"\t"+
                                        str(round(self.channels[13].get(), 2))+"\t"+
                                        str(round(self.channels[14].get(), 2))+"\t"+
                                        str(round(self.channels[15].get(), 2))+"\t"+
                                        str(round(self.channels[16].get(), 2))+"\t"+
                                        str(round(self.channels[17].get(), 2))+"\t"+
                                        str(round(self.channels[18].get(), 2))+"\t"+
                                        str(round(self.channels[19].get(), 2))+"\t")
                except curses.error:
                    pass
                try:
                    window.addstr(17,0, str(round(self.channels[20].get(), 2))+"\t"+
                                        str(round(self.channels[21].get(), 2))+"\t"+
                                        str(round(self.channels[22].get(), 2))+"\t"+
                                        str(round(self.channels[23].get(), 2))+"\t"+
                                        str(round(self.channels[24].get(), 2))+"\t"+
                                        str(round(self.channels[25].get(), 2))+"\t"+
                                        str(round(self.channels[26].get(), 2))+"\t"+
                                        str(round(self.channels[27].get(), 2))+"\t"+
                                        str(round(self.channels[28].get(), 2))+"\t"+
                                        str(round(self.channels[29].get(), 2))+"\t")
                except curses.error:
                    pass
                try:
                    window.addstr(18,0, str(round(self.channels[30].get(), 2))+"\t"+
                                        str(round(self.channels[31].get(), 2))+"\t"+
                                        str(round(self.channels[32].get(), 2))+"\t"+
                                        str(round(self.channels[33].get(), 2))+"\t"+
                                        str(round(self.channels[34].get(), 2))+"\t"+
                                        str(round(self.channels[35].get(), 2))+"\t"+
                                        str(round(self.channels[36].get(), 2))+"\t"+
                                        str(round(self.channels[37].get(), 2))+"\t"+
                                        str(round(self.channels[38].get(), 2))+"\t"+
                                        str(round(self.channels[39].get(), 2))+"\t")
                except curses.error:
                    pass
                try:
                    window.addstr(19,0, str(round(self.channels[40].get(), 2))+"\t"+
                                        str(round(self.channels[41].get(), 2))+"\t"+
                                        str(round(self.channels[42].get(), 2))+"\t"+
                                        str(round(self.channels[43].get(), 2))+"\t"+
                                        str(round(self.channels[44].get(), 2))+"\t"+
                                        str(round(self.channels[45].get(), 2))+"\t"+
                                        str(round(self.channels[46].get(), 2))+"\t"+
                                        str(round(self.channels[47].get(), 2))+"\t"+
                                        str(round(self.channels[48].get(), 2))+"\t"+
                                        str(round(self.channels[49].get(), 2))+"\t"+
                                        str(round(self.channels[50].get(), 2))+"\t")
                except curses.error:
                    pass

                    
                try:
                    window.refresh()
                    time.sleep(0.001)
                except curses.error:
                    pass

            score = self.update_score()
            print(score)
            self.c.readScore(score)
            #self.c.rewindScore()
            res = self.c.performBuffer()
            while res == 0:
                res = self.c.performBuffer()
                time.sleep(0.01)
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
        ratio_fac = params[0] * 8.
        params[2] *= 440. * 4.
        params[2] += 110.
        #params[0] *= 60                # fr
        #params[0] += 12
        #params[0] = 440. * np.power(2, (round(params[0])-49) / 12.)
        params[2:32] *= adsr_fac
        params[32:42] *= 1.             # amp
        params[42:52] *= ratio_fac
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
        #for param in self.params:
        #    out += " "+str(param)
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

	icps	    chnget  "param1"
	kfund		portk	icps, kporttime		;SMOOTH VARIABLE CHANGES WITH PORTK
	kamp	=	0.1

    kampatt1	init	0
	kampatt2	init	0
	kampatt3	init	0
	kampatt4	init	0
	kampatt5	init	0
	kampatt6	init	0
	kampatt7	init	0
	kampatt8	init	0
	kampatt9	init	0
	kampatt10	init	0
	kampdec1	init	0
	kampdec2	init	0
	kampdec3	init	0
	kampdec4	init	0
	kampdec5	init	0
	kampdec6	init	0
	kampdec7	init	0
	kampdec8	init	0
	kampdec9	init	0
	kampdec10	init	0
	kamprel1	init	0
	kamprel2	init	0
	kamprel3	init	0
	kamprel4	init	0
	kamprel5	init	0
	kamprel6	init	0
	kamprel7	init	0
	kamprel8	init	0
	kamprel9	init	0
	kamprel10	init	0

	kPartAmp1	init	0
	kPartAmp2	init	0
	kPartAmp3	init	0
	kPartAmp4	init	0
	kPartAmp5	init	0
	kPartAmp6	init	0
	kPartAmp7	init	0
	kPartAmp8	init	0
	kPartAmp9	init	0
	kPartAmp10	init	0

	kratio1		init	0
	kratio2		init	0
	kratio3		init	0
	kratio4		init	0
	kratio5		init	0
	kratio6		init	0
	kratio7		init	0
	kratio8		init	0
	kratio9		init	0
	kratio10	init	0

	kampatt1	chnget	"param5"
	kampatt2	chnget	"param6"
	kampatt3	chnget	"param5"
	kampatt4	chnget	"param7"
	kampatt5	chnget	"param8"
	kampatt6	chnget	"param9"
	kampatt7	chnget	"param10"
	kampatt8	chnget	"param11"
	kampatt9	chnget	"param12"
	kampatt10	chnget	"param13"
	kampdec1	chnget	"param14"
	kampdec2	chnget	"param15"
	kampdec3	chnget	"param16"
	kampdec4	chnget	"param17"
	kampdec5	chnget	"param18"
	kampdec6	chnget	"param19"
	kampdec7	chnget	"param20"
	kampdec8	chnget	"param21"
	kampdec9	chnget	"param22"
	kampdec10	chnget	"param23"
	kamprel1	chnget	"param24"
	kamprel2	chnget	"param25"
	kamprel3	chnget	"param26"
	kamprel4	chnget	"param27"
	kamprel5	chnget	"param28"
	kamprel6	chnget	"param29"
	kamprel7	chnget	"param30"
	kamprel8	chnget	"param31"
	kamprel9	chnget	"param32"
	kamprel10	chnget	"param33"

	kPartAmp1	chnget	"param34"
	kPartAmp2	chnget	"param35"
	kPartAmp3	chnget	"param36"
	kPartAmp4	chnget	"param37"
	kPartAmp5	chnget	"param38"
	kPartAmp6	chnget	"param39"
	kPartAmp7	chnget	"param40"
	kPartAmp8	chnget	"param41"
	kPartAmp9	chnget	"param42"
	kPartAmp10	chnget	"param43"

	kratio1		chnget	"param44"
	kratio2		chnget	"param45"
	kratio3		chnget	"param46"
	kratio4		chnget	"param47"
	kratio5		chnget	"param48"
	kratio6		chnget	"param49"
	kratio7		chnget	"param50"
	kratio8		chnget	"param51"
	kratio9		chnget	"param52"
	kratio10	chnget	"param53"
	
	

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