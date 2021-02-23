<CsoundSynthesizer>
<CsOptions>
-Ma
</CsOptions>
<CsInstruments>
sr=8000
ksmps=32
nchnls=1
0dbfs=1

#define R_PORT      # 7770 #

gisine	ftgen 0, 0, 2^13, 10, 1
gihandle OSCinit $R_PORT

turnon 1000

instr 	1000
	kPorttime	linseg	0,0.01,0.05

    kP1 init 0
    kParam1 OSClisten gihandle,        "/tensaudio/param1", "f", kP1
    kP2 init 0
    kParam2 OSClisten gihandle,        "/tensaudio/param2", "f", kP2
    kP3 init 0
    kParam3 OSClisten gihandle,        "/tensaudio/param3", "f", kP3
    kP4 init 1
    kParam4 OSClisten gihandle,        "/tensaudio/param4", "f", kP4
    kP5 init 1
    kParam5 OSClisten gihandle,        "/tensaudio/param5", "f", kP5
    kP6 init 0
    kParam6 OSClisten gihandle,        "/tensaudio/param6", "f", kP6
    kP7 init 0
    kParam7 OSClisten gihandle,        "/tensaudio/param7", "f", kP7
    kP8 init 0
    kParam8 OSClisten gihandle,        "/tensaudio/param8", "f", kP8
    kP9 init 0
    kParam9 OSClisten gihandle,        "/tensaudio/param9", "f", kP9
    kP10 init 0
    kParam10 OSClisten gihandle,        "/tensaudio/param10", "f", kP10
    kP11 init 0
    kParam11 OSClisten gihandle,        "/tensaudio/param11", "f", kP11
    kP12 init 0
    kParam12 OSClisten gihandle,        "/tensaudio/param12", "f", kP12
    kP13 init 0
    kParam13 OSClisten gihandle,        "/tensaudio/param13", "f", kP13
    kP14 init 0
    kParam14 OSClisten gihandle,        "/tensaudio/param14", "f", kP14
    kP15 init 0
    kParam15 OSClisten gihandle,        "/tensaudio/param15", "f", kP15
    kP16 init 0
    kParam16 OSClisten gihandle,        "/tensaudio/param16", "f", kP16
    kP17 init 0
    kParam17 OSClisten gihandle,        "/tensaudio/param17", "f", kP17
    kP18 init 0
    kParam18 OSClisten gihandle,        "/tensaudio/param18", "f", kP18
    kP19 init 0
    kParam19 OSClisten gihandle,        "/tensaudio/param19", "f", kP19
    kP20 init 0
    kParam20 OSClisten gihandle,        "/tensaudio/param20", "f", kP20
    kP21 init 0
    kParam21 OSClisten gihandle,        "/tensaudio/param21", "f", kP21
    kP22 init 0
    kParam22 OSClisten gihandle,        "/tensaudio/param22", "f", kP22
    kP23 init 0
    kParam23 OSClisten gihandle,        "/tensaudio/param23", "f", kP23
    kP24 init 0
    kParam24 OSClisten gihandle,        "/tensaudio/param24", "f", kP24
    kP25 init 0
    kParam25 OSClisten gihandle,        "/tensaudio/param25", "f", kP25
    kP26 init 0
    kParam26 OSClisten gihandle,        "/tensaudio/param26", "f", kP26
    kP27 init 0
    kParam27 OSClisten gihandle,        "/tensaudio/param27", "f", kP27
    kP28 init 0
    kParam28 OSClisten gihandle,        "/tensaudio/param28", "f", kP28
    kP29 init 0
    kParam29 OSClisten gihandle,        "/tensaudio/param29", "f", kP29
    kP30 init 0
    kParam30 OSClisten gihandle,        "/tensaudio/param30", "f", kP30
    kP31 init 0
    kParam31 OSClisten gihandle,        "/tensaudio/param31", "f", kP31
    kP32 init 0
    kParam32 OSClisten gihandle,        "/tensaudio/param32", "f", kP32
    kP33 init 0
    kParam33 OSClisten gihandle,        "/tensaudio/param33", "f", kP33
    kP34 init 0
    kParam34 OSClisten gihandle,        "/tensaudio/param34", "f", kP34
    kP35 init 0
    kParam35 OSClisten gihandle,        "/tensaudio/param35", "f", kP35
    kP36 init 0
    kParam36 OSClisten gihandle,        "/tensaudio/param36", "f", kP36
    kP37 init 0
    kParam37 OSClisten gihandle,        "/tensaudio/param37", "f", kP37
    kP38 init 0
    kParam38 OSClisten gihandle,        "/tensaudio/param38", "f", kP38
    kP39 init 0
    kParam39 OSClisten gihandle,        "/tensaudio/param39", "f", kP39
    kP40 init 0
    kParam40 OSClisten gihandle,        "/tensaudio/param40", "f", kP40
    kP41 init 0
    kParam41 OSClisten gihandle,        "/tensaudio/param41", "f", kP41
    kP42 init 0
    kParam42 OSClisten gihandle,        "/tensaudio/param42", "f", kP42
    kP43 init 0
    kParam43 OSClisten gihandle,        "/tensaudio/param43", "f", kP43
    kP44 init 0
    kParam44 OSClisten gihandle,        "/tensaudio/param44", "f", kP44
    kP45 init 0
    kParam45 OSClisten gihandle,        "/tensaudio/param45", "f", kP45
    kP46 init 0
    kParam46 OSClisten gihandle,        "/tensaudio/param46", "f", kP46
    kP47 init 0
    kParam47 OSClisten gihandle,        "/tensaudio/param47", "f", kP47
    kP48 init 0
    kParam48 OSClisten gihandle,        "/tensaudio/param48", "f", kP48
    kP49 init 0
    kParam49 OSClisten gihandle,        "/tensaudio/param49", "f", kP49
    kP50 init 0
    kParam50 OSClisten gihandle,        "/tensaudio/param50", "f", kP50
    kP51 init 0
    kParam51 OSClisten gihandle,        "/tensaudio/param51", "f", kP51
    kP52 init 0
    kParam52 OSClisten gihandle,        "/tensaudio/param52", "f", kP52
    kP53 init 0
    kParam53 OSClisten gihandle,        "/tensaudio/param53", "f", kP53
    kP54 init 0
    kParam54 OSClisten gihandle,        "/tensaudio/param54", "f", kP54
    kP55 init 0
    kParam55 OSClisten gihandle,        "/tensaudio/param55", "f", kP55
    kP56 init 0
    kParam56 OSClisten gihandle,        "/tensaudio/param56", "f", kP56
    kP57 init 0
    kParam57 OSClisten gihandle,        "/tensaudio/param57", "f", kP57
    kP58 init 0
    kParam58 OSClisten gihandle,        "/tensaudio/param58", "f", kP58
    kP59 init 0
    kParam59 OSClisten gihandle,        "/tensaudio/param59", "f", kP59
    kP60 init 0
    kParam60 OSClisten gihandle,        "/tensaudio/param60", "f", kP60
    kP61 init 0
    kParam61 OSClisten gihandle,        "/tensaudio/param61", "f", kP61
    kP62 init 0
    kParam62 OSClisten gihandle,        "/tensaudio/param62", "f", kP62
    kP63 init 0
    kParam63 OSClisten gihandle,        "/tensaudio/param63", "f", kP63
    kP64 init 0
    kParam64 OSClisten gihandle,        "/tensaudio/param64", "f", kP64

    kcps	    =       kP1
	kfund		portk	kcps, kPorttime
	kamp	=	1

	kampatt1	= kP2
	kampatt2	= kP3
	
	kampdec1	= kP4
	kampdec2	= kP5
	
	kamprel1	= kP6
	kamprel2	= kP7
	

	kPartAmp1	= kP8
	kPartAmp2	= kP9
	

	kratio1		= kP10
	kratio2		= kP11
	

	kenv1		expsegr	.001, i(kampatt1), 1.001, i(kampdec1), .001, i(kamprel1), .001
	kenv2		expsegr	.001, i(kampatt2), 1.001, i(kampdec2), .001, i(kamprel2), .001
	

	; apart1		foscil	kamp*(aenv1  - 0.001), kfund,1, kratio1, (1*kPartAmp1), gisine
    ; apart2		foscil	kamp*(aenv2  - 0.001), kfund,1, kratio2, (1*kPartAmp2), gisine
    apart1		foscil	kenv1*kamp, kfund,1, kratio1, (kPartAmp1), gisine
    apart2		foscil	kenv2*kamp, kfund,1, kratio2, (kPartAmp2), gisine
	amix		sum apart1, apart2

	out		    amix
endin


</CsInstruments>
<CsScore>
f 0 0.5
e

</CsScore>
</CsoundSynthesizer>