RESOURCES_DIR = "C:\\tensaudio_resources\\"

PLOTS_DIR = "D:\\tensaudio_plots\\"
TRAINING_DIR = "D:\\tensaudio_training\\"

MODEL_DIR = "D:\\tensaudio_models\\"

DATASET_DIRNAME = "piano/ta"

# must be divisible by 100
VIS_WIDTH = 1600
VIS_HEIGHT = 900

VIS_UPDATE_INTERVAL = 1 # iterations

VIS_N_FFT = 2048
VIS_HOP_LEN = VIS_N_FFT//4

SLEEP_TIME = 0
MAX_ITERS_PER_SEC = 0

SAVE_EVERY_EPOCH = 50
SAVE_EVERY_BATCHES = 54

VERBOSITY_LEVEL = 1 # 0, 1, 2

GRIFFIN_LIM_MAX_ITERS_PREVIEW = 0
GRIFFIN_LIM_MAX_ITERS_SAVING = 0

# Hyperparameters
GEN_LR = 0.0004
DIS_LR = 0.0002
BETA = 0.5
GEN_MOMENTUM = 0.01

BATCH_SIZE = 32 # lower = faster but lower quality training (must be 2 or greater)

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!
GEN_MODE = 6            # 0 = TBI
                        # 1 = TBI
                        # 2 = TBI
                        # 3 = Conv/Audio mode
                        # 4 = Conv/Mel mode (WIP)
                        # 5 = Conv/STFT "Specgram" mode ([batch, 2, bin, frame])
                        # 6 = Conv/Hilbert "Specgram" mode ([batch, 2, time])
                        # 10 = CSound Synthesizer mode
USE_REAL_AUDIO = False
SAMPLE_RATE = 22050
GEN_SAMPLE_RATE_FACTOR = 1
SUBTYPE = 'PCM_16'
INPUT_DURATION = 100 / SAMPLE_RATE
OUTPUT_DURATION = 5

GEN_INITIAL_LIN_SCALE = 1          # higher = more memory, must be 1 or greater
GEN_MAX_LIN_FEATURES = 4096
GEN_KERNEL_SIZE_UPSCALING = 53    # higher = more memory, supposedly odd numbers work better
GEN_STRIDE_UPSCALING = 2      # higher = more memory, must be greater than GEN_STRIDE_DOWNSCALING
GEN_KERNEL_SIZE_DOWNSCALING = 53   # higher = more memory, supposedly odd numbers work better
GEN_STRIDE_DOWNSCALING = 1          # must be 1 or greater
GEN_MIN_LAYERS = 1
GEN_MAX_CHANNELS = 4


# Audio mode only
N_CHANNELS = 1

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

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/dis_ckpts folder or the discsriminator model will give
# an error!

DIS_MODE = 3            # 0 = Direct mode
                        # 1 = FFT mode
                        # 2 = Mel mode
                        # 3 = Specgram mode
REAL_LABEL = 1.
FAKE_LABEL = 0.
PHASE_SHUFFLE = 0.03
PHASE_SHUFFLE_CHANCE = 0.0
DIS_DROPOUT = 0.00
DIS_MAX_CHANNELS = 32
DIS_STRIDE = 2
DIS_KERNEL_SIZE = 1
DIS_N_FFT = VIS_N_FFT
DIS_N_MELS = N_GEN_MEL_CHANNELS
DIS_HOP_LEN = VIS_HOP_LEN







# DO NOT CHANGE ANYTHING BELOW THIS LINE!
# ---------------------------------------------------------------------------------

import numpy as np
import librosa

EPSILON = 1e-9

TOTAL_SAMPLES_IN = int(SAMPLE_RATE * INPUT_DURATION)
TOTAL_SAMPLES_OUT = int(SAMPLE_RATE * OUTPUT_DURATION)
#N_BATCHES = int(TOTAL_SAMPLES_OUT // BATCH_SIZE)
#if TOTAL_SAMPLES_OUT % BATCH_SIZE != 0:
#    raise ValueError("Could not calculate N_BATCHES: Total length of audio not divisible by", (BATCH_SIZE))
#N_TIMESTEPS_PER_KERNEL = int(SAMPLE_RATE*OUTPUT_DURATION // (GEN_KERNEL_SIZE_DOWNSCALING * N_BATCHES))
#SAMPLES_PER_BATCH = int(TOTAL_SAMPLES_OUT // N_BATCHES)
#if TOTAL_SAMPLES_OUT % N_BATCHES != 0:
#    raise ValueError("Could not calculate SAMPLES_PER_BATCH: Total length of audio not divisible by", N_BATCHES)
KONTROL_SECONDS = KONTROL_SAMPLES/SAMPLE_RATE

GEN_N_FRAMES = librosa.samples_to_frames(TOTAL_SAMPLES_OUT, GEN_HOP_LEN, N_GEN_FFT)

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
    output_samples = N_BATCHES*N_CHANNELS*N_TIMESTEPS_PER_KERNEL*GEN_KERNEL_SIZE_DOWNSCALING
    print("Total # of output samples:", output_samples)
    print("^-"*39 + "^")

def v_cprint(*s):
    if VERBOSITY_LEVEL >= 1:
        print(*s)
def vv_cprint(*s):
    if VERBOSITY_LEVEL >= 2:
        print(*s)