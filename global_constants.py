RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "fire"
EXAMPLE_RESULTS_DIR = "synthloops"
INPUTS_DIR = "inputs_kicks"

PLOTS_DIR = "D:\\tensaudio_plots"
TRAINING_DIR = "D:\\tensaudio_training"

MODEL_DIR = "D:\\tensaudio_models"

# must be divisible by 100
VIS_WIDTH = 1200
VIS_HEIGHT = 700
VIS_UPDATE_INTERVAL = 25 # iterations

VIS_N_FFT = 512
VIS_HOP_LEN = 1

# set to 0 to run until visualizer is closed
MAX_ITERS = 0
RUN_FOR_SEC = 0

SLEEP_TIME = 0
MAX_ITERS_PER_SEC = 0

# set to 0 to disable periodically generating progress updates
SAVE_EVERY_SECONDS = 0
# set to 0 to disable periodically saving model
SAVE_MODEL_EVERY_ITERS = 10000

VERBOSITY_LEVEL = 1 # 0, 1, 2

GRIFFIN_LIM_MAX_ITERS_PREVIEW = 1
GRIFFIN_LIM_MAX_ITERS_SAVING = 16

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!
GEN_MODE = 4            # 0 = RNN/Hilbert mode
                        # 1 = RNN/Audio mode
                        # 2 = Conv/Hilbert mode
                        # 3 = Conv/Audio mode
                        # 4 = Conv/Mel mode
                        # 5 = Conv/STFT mode
                        # 10 = CSound Synthesizer mode
USE_REAL_AUDIO = False
SAMPLE_RATE = 22050
GEN_SAMPLE_RATE_FACTOR = 1
SUBTYPE = 'PCM_16'
INPUT_DURATION = 32 / SAMPLE_RATE
OUTPUT_DURATION = 2**16 / SAMPLE_RATE # power of 2 samples
GEN_KERNEL_SIZE = 1    # higher = more memory
GEN_STRIDE1 = 2         # higher = more memory
#GEN_STRIDE2 = 1         # higher = more memory
#GEN_SCALE1 = 3          # higher = more memory, must be 2 or greater
GEN_SCALE = 2        # higher = more memory, must be 2 or greater
MIN_N_GEN_LAYERS = 1
# Non-Mel mode only
N_CHANNELS = 4
# Mel mode only
N_GEN_MEL_CHANNELS = 128
# Hilbert, STFT, and Mel mode only
N_GEN_FFT = VIS_N_FFT
GEN_HOP_LEN = VIS_HOP_LEN
# RNN mode only
N_RNN_LAYERS = 4
# CSound mode only
N_PARAMS = 64
KONTROL_SAMPLES = 64
PARAM_UPDATE_SAMPLES = SAMPLE_RATE*OUTPUT_DURATION
TOTAL_PARAM_UPDATES = SAMPLE_RATE*OUTPUT_DURATION//PARAM_UPDATE_SAMPLES
# Non-CSound mode only
BATCH_SIZE = 1 # also a power of 2, lower = more efficient but lower quality

# Hyperparameters
GENERATOR_LR = 0.0001
GENERATOR_BETA = 0.5
#GENERATOR_MOMENTUM = 0.02

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/dis_ckpts folder or the discsriminator model will give
# an error!
INPUT_MODE = 'direct'   # 'direct' = direct comparison of example and example result
                        # 'conv' = comparison of example and convolved example result
DIS_MODE = 2            # 0 = Direct mode
                        # 1 = FFT mode
                        # 2 = Mel mode
                        # 3 = Hilbert mode
REAL_LABEL = 1.
FAKE_LABEL = 0.
N_DIS_LAYERS = 2
DIS_STRIDE = 16
DIS_KERNEL_SIZE = 1
DIS_N_FFT = N_GEN_FFT
#DIS_HOP_LEN = 64
DIS_N_MELS = N_GEN_MEL_CHANNELS
DIS_HOP_LEN = GEN_HOP_LEN
#DIS_FFT_VAL = N_GEN_FFT

# Hyperparameters
DISCRIMINATOR_LR = 0.0001
DISCRIMINATOR_BETA = 0.5
#DISCRIMINATOR_MOMENTUM = 0.2







# DO NOT CHANGE ANYTHING BELOW THIS LINE!
# ---------------------------------------------------------------------------------

import numpy as np
import curses

TOTAL_SAMPLES_IN = int(SAMPLE_RATE * INPUT_DURATION)
TOTAL_SAMPLES_OUT = int(SAMPLE_RATE * OUTPUT_DURATION)
N_BATCHES = int(TOTAL_SAMPLES_OUT // BATCH_SIZE)
if TOTAL_SAMPLES_OUT % BATCH_SIZE != 0:
    raise ValueError("Could not calculate N_BATCHES: Total length of audio not divisible by", (BATCH_SIZE))
N_TIMESTEPS_PER_KERNEL = int(SAMPLE_RATE*OUTPUT_DURATION // (GEN_KERNEL_SIZE * N_BATCHES))
SAMPLES_PER_BATCH = int(TOTAL_SAMPLES_OUT // N_BATCHES)
if TOTAL_SAMPLES_OUT % N_BATCHES != 0:
    raise ValueError("Could not calculate SAMPLES_PER_BATCH: Total length of audio not divisible by", N_BATCHES)
KONTROL_SECONDS = KONTROL_SAMPLES/SAMPLE_RATE

def print_global_constants():
    print("v-"*39 + "v")
    print("Total # of input samples:", TOTAL_SAMPLES_IN)
    print("Timesteps per kernel:", N_TIMESTEPS_PER_KERNEL)
    print("Batches per layer:", N_BATCHES)
    print("Samples per batch:", SAMPLES_PER_BATCH)
    print("Number of channels:", N_CHANNELS)
    # if GEN_MODE == 1:
    #     cprint("Will create", N_RNN_LAYERS, "layers of", N_UNITS, "units.")
    # elif GEN_MODE == 0:
    #     cprint("Will create 1 Pre-Dense layer of", SAMPLES_PER_BATCH, "filters.")    
    #     if N_PRE_DENSE_LAYERS > 1:
    #         cprint("Will create", N_PRE_DENSE_LAYERS-1, "Pre-Dense layers of", N_UNITS*N_TIMESTEPS, "filters.")
    #     cprint("Will create", N_DENSE_LAYERS-1, "Dense layers of", SAMPLES_PER_BATCH, "units.")
    #     cprint("Will create 1 Dense layer of", SAMPLES_PER_BATCH, "units.")
    #     if N_POST_DENSE_LAYERS > 1:
    #         for i in range(1,N_POST_DENSE_LAYERS):
    #                 n_filts = i*N_POST_DENSE_FILTERS
    #                 cprint("Will create 1 Post-Dense layer of", n_filts, "filters.")
    #     else:
    #         cprint("Will create 1 Post-Dense layer of", N_UNITS*N_TIMESTEPS//20, "filters.")
    output_samples = N_BATCHES*N_CHANNELS*N_TIMESTEPS_PER_KERNEL*GEN_KERNEL_SIZE
    print("Total # of output samples:", output_samples)
    print("^-"*39 + "^")

def v_cprint(*s):
    if VERBOSITY_LEVEL >= 1:
        print(*s)
def vv_cprint(*s):
    if VERBOSITY_LEVEL >= 2:
        print(*s)