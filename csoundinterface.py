import time
import ctcsound
import torch
import librosa
from pythonosc.udp_client import SimpleUDPClient
from global_constants import *

class CsoundInterface():
    def __init__(self, vis):
        self.param_len_in_seconds = OUTPUT_DURATION/TOTAL_PARAM_UPDATES
        self.c = ctcsound.Csound()

        self.client = SimpleUDPClient("127.0.0.1", 7770)
        self.vis = vis
        self.total_instruments = 1
        self.total_gens = 1
        self.params = [1.] * N_PARAMS
        # self.channels = [
        #     ChannelUpdater(self.c, "param"+str(i), self.yield_param(i)) for i in range(1, N_PARAMS+1)
        # ]
    
    def set_param(self, num, value):
        self.params[num-1] = value
        self.client.send_message("/tensaudio/param"+str(num), float(value))
        #print("/param"+str(num)+", "+ str(float(value)))
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

    def stop(self):
        self.c.stop()
        self.c.cleanup()
    def reset(self):
        self.c.reset()
    def compile(self):
        #self.c.setOption('-Ma')
        result = self.c.compileCsd("./tensaudio.csd")
        if result != 0:
            return result
        self.c.start()
        #self.c.readScore(self.update_score())
        return 0

    def perform(self, params, window):
        out = np.array([])
        done = False
        last_params = params[0:N_PARAMS]
        self.update_params(last_params)

        i = 1
        tim = ctcsound.RtClock()
        self.c.initTimerStruct(tim)
        tim_cpu = self.c.CPUTime(tim)
        while not done:
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
            """
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
                    window.addstr(15,0, str(round(self.params[0], 2))+"\t"+
                                        str(round(self.params[1], 2))+"\t"+
                                        str(round(self.params[2], 2))+"\t"+
                                        str(round(self.params[3], 2))+"\t"+
                                        str(round(self.params[4], 2))+"\t"+
                                        str(round(self.params[5], 2))+"\t"+
                                        str(round(self.params[6], 2))+"\t"+
                                        str(round(self.params[7], 2))+"\t"+
                                        str(round(self.params[8], 2))+"\t"+
                                        str(round(self.params[9], 2))+"\t")
                except curses.error:
                    pass
                try:
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
                except curses.error:
                    pass
                try:
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
                except curses.error:
                    pass
                try:
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
                except curses.error:
                    pass
                try:
                    window.addstr(19,0, str(round(self.params[40], 2))+"\t"+
                                        str(round(self.params[41], 2))+"\t"+
                                        str(round(self.params[42], 2))+"\t"+
                                        str(round(self.params[43], 2))+"\t"+
                                        str(round(self.params[44], 2))+"\t"+
                                        str(round(self.params[45], 2))+"\t"+
                                        str(round(self.params[46], 2))+"\t"+
                                        str(round(self.params[47], 2))+"\t"+
                                        str(round(self.params[48], 2))+"\t"+
                                        str(round(self.params[49], 2))+"\t"+
                                        str(round(self.params[50], 2))+"\t")
                except curses.error:
                    pass

                try:
                    window.refresh()
                    time.sleep(0.001)
                except curses.error:
                    pass

            """
            done2 = False
            while not done2:
                res = 0
                while not res:
                    res = self.c.performBuffer()
                    newtime = self.c.CPUTime(tim)
                    if res > 0 or (newtime - tim_cpu) >= self.param_len_in_seconds:
                        break
                    self.update_params(param_slice)
                out = np.concatenate((out, self.c.outputBuffer()))
                done2 = len(out) >= self.param_len_in_seconds*SAMPLE_RATE
            done = len(out) >= TOTAL_SAMPLES_OUT
            i += 1
        return out

    def scale_params(self, params):
        params = params.flatten()
        if len(params) != N_PARAMS:
            raise ValueError("Incorrect number of parameters!")
        adsr_fac = self.param_len_in_seconds * 4
        params[0] *= 440. * 4.
        params[0] += 110.
        ratio_fac = params[13] * 8.
        params[2:7] *= adsr_fac
        params[10:11] *= ratio_fac
        return params

    def update_params(self, params):
        params = params.flatten()
        if len(params) != N_PARAMS:
            raise ValueError("Incorrect number of parameters!")
        for i in range(N_PARAMS):
            #print(str(i)+": "+str(params[i]))
            self.set_param(i+1, params[i])
        self.vis.update_vals()
        self.vis.update_display()
    
    def update_score(self):
        out = """\n
        r60
        f 0 3600
        s
        e"""
        #for param in self.params:
        #    out += " "+str(param)
        out += "\n"
        return out

FM_SYNTH = """
instr 	1000
	kPorttime	linseg	0,0.01,0.05		;PORTAMENTO TIME RAMPS UP QUICKLY FROM ZERO

    

"""
param_init_string = ""
for i in range(1,N_PARAMS+1):
    param_init_string += "kP"+str(i)+" init 0\n"
    param_init_string += "kParam"+str(i)+" OSClisten gihandle,        \"/tensaudio/param"+str(i)+"\", \"f\", kP"+str(i)+"\n"
FM_SYNTH += param_init_string + """

    kcps	    =       kP51
	kfund		portk	kcps, kPorttime		;SMOOTH VARIABLE CHANGES WITH PORTK
	kamp	=	1

	kampatt1	= kP1
	kampatt2	= kP2
	kampatt3	= kP3
	kampatt4	= kP4
	kampatt5	= kP5
	kampatt6	= kP6
	kampatt7	= kP7
	kampatt8	= kP8
	kampatt9	= kP9
	kampatt10	= kP10
	kampdec1	= kP11
	kampdec2	= kP12
	kampdec3	= kP13
	kampdec4	= kP14
	kampdec5	= kP15
	kampdec6	= kP16
	kampdec7	= kP17
	kampdec8	= kP18
	kampdec9	= kP19
	kampdec10	= kP20
	kamprel1	= kP21
	kamprel2	= kP22
	kamprel3	= kP23
	kamprel4	= kP24
	kamprel5	= kP25
	kamprel6	= kP26
	kamprel7	= kP27
	kamprel8	= kP28
	kamprel9	= kP29
	kamprel10	= kP30

	kPartAmp1	= kP31
	kPartAmp2	= kP32
	kPartAmp3	= kP33
	kPartAmp4	= kP34
	kPartAmp5	= kP35
	kPartAmp6	= kP36
	kPartAmp7	= kP37
	kPartAmp8	= kP38
	kPartAmp9	= kP39
	kPartAmp10	= kP40

	kratio1		= kP41
	kratio2		= kP42
	kratio3		= kP43
	kratio4		= kP44
	kratio5		= kP45
	kratio6		= kP46
	kratio7		= kP47
	kratio8		= kP48
	kratio9		= kP49
	kratio10	= kP50
	
	

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