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

SAVE_EVERY_EPOCH = 10
SAVE_EVERY_BATCHES = 54

VERBOSITY_LEVEL = 1 # 0, 1, 2

GRIFFIN_LIM_MAX_ITERS_PREVIEW = 0
GRIFFIN_LIM_MAX_ITERS_SAVING = 0

# Hyperparameters
GEN_LR = 0.0004
DIS_LR = 0.0004
BETA = 0.0
#GEN_MOMENTUM = 0.01

BATCH_SIZE = 35 # lower = faster but lower quality training (must be 2 or greater)
N_CRITIC = 1
N_GEN = 3

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
INPUT_DURATION = 256 / SAMPLE_RATE
OUTPUT_DURATION = 2

GEN_INITIAL_LIN_SCALE = 1          # higher = more memory, must be 1 or greater
GEN_MAX_LIN_FEATURES = 4096
GEN_KERNEL_SIZE_UPSCALING = 53    # higher = more memory, supposedly odd numbers work better
GEN_STRIDE_UPSCALING = 2      # higher = more memory, must be greater than GEN_STRIDE_DOWNSCALING
GEN_KERNEL_SIZE_DOWNSCALING = 53   # higher = more memory, supposedly odd numbers work better
GEN_STRIDE_DOWNSCALING = 1          # must be 1 or greater
GEN_MIN_LAYERS = 1
GEN_MAX_CHANNELS = 4
GEN_DROPOUT = 0.2


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
REAL_LABEL = -1.
FAKE_LABEL = 1.
PHASE_SHUFFLE = 0.03
PHASE_SHUFFLE_CHANCE = 0.1
DIS_DROPOUT = 0.2
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

KONTROL_SECONDS = KONTROL_SAMPLES/SAMPLE_RATE

GEN_N_FRAMES = librosa.samples_to_frames(TOTAL_SAMPLES_OUT, GEN_HOP_LEN, N_GEN_FFT)

def v_cprint(*s):
    if VERBOSITY_LEVEL >= 1:
        print(*s)
def vv_cprint(*s):
    if VERBOSITY_LEVEL >= 2:
        print(*s)