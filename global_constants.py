RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "kicks"
EXAMPLE_RESULTS_DIR = "hihats"
INPUTS_DIR = "inputs_kicks"

PLOTS_DIR = "D:\\tensaudio_plots"
TRAINING_DIR = "D:\\tensaudio_training"

MODEL_DIR = "D:\\tensaudio_models"

# set to 0 to run until Ctrl+C is input in the terminal
MAX_ITERS = 0

SLEEP_TIME = 0

# set to 0 to disable periodically generating progress updates
SAVE_EVERY_SECONDS = 30
# set to 0 to disable periodically saving model
SAVE_MODEL_EVERY_SECONDS = 10*60

VERBOSITY_LEVEL = 1 # 0, 1, 2

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!
INPUT_MODE = 'direct'   # 'direct' = direct comparison of example and example result
                        # 'conv' = comparison of example and convolved example result

GEN_MODE = 3            # 0 = RNN/Hilbert mode
                        # 1 = RNN/Audio mode
                        # 2 = Conv/Hilbert mode
                        # 3 = Conv/Audio mode
USE_REAL_AUDIO = False
SAMPLE_RATE = 8000
SUBTYPE = 'PCM_16'
INPUT_DURATION = 8 / SAMPLE_RATE
OUTPUT_DURATION = 2
N_RNN_LAYERS = 4
DESIRED_PROCESS_UNITS = 512
N_PROCESS_LAYERS = 64
BATCH_OPTIMIZATION_FACTOR = 4000
GEN_KERNEL_SIZE = 1
N_TIMESTEPS_PER_KERNEL = SAMPLE_RATE*OUTPUT_DURATION // (GEN_KERNEL_SIZE * 2)
GENERATOR_LR = 0.01
GENERATOR_BETA = 0.5

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/dis_ckpts folder or the discsriminator model will give
# an error!
REAL_LABEL = 1.
FAKE_LABEL = 0.
N_FFT = 2048
N_DIS_LAYERS = 16
N_DIS_CHANNELS = SAMPLE_RATE//80
DIS_KERNEL_SIZE = 1
DISCRIMINATOR_LR = 0.002
DISCRIMINATOR_BETA = 0.5

# DO NOT CHANGE ANYTHING BELOW THIS LINE!
# ---------------------------------------------------------------------------------

import numpy as np


TOTAL_SAMPLES_IN = int(SAMPLE_RATE * INPUT_DURATION)
TOTAL_SAMPLES_OUT = int(SAMPLE_RATE * OUTPUT_DURATION)
if GEN_MODE in [0, 2]:
    N_CHANNELS = 2
else:
    N_CHANNELS = 1
N_BATCHES = TOTAL_SAMPLES_OUT // (N_CHANNELS * GEN_KERNEL_SIZE * N_TIMESTEPS_PER_KERNEL)
if BATCH_OPTIMIZATION_FACTOR * TOTAL_SAMPLES_OUT % (N_CHANNELS * GEN_KERNEL_SIZE * N_TIMESTEPS_PER_KERNEL) != 0:
    raise ValueError("Could not calculate N_BATCHES: Total length of audio not divisible by", (N_CHANNELS * GEN_KERNEL_SIZE * N_TIMESTEPS_PER_KERNEL))
SAMPLES_PER_BATCH = TOTAL_SAMPLES_OUT // N_BATCHES
if TOTAL_SAMPLES_OUT % N_BATCHES != 0:
    raise ValueError("Could not calculate SAMPLES_PER_BATCH: Total length of audio not divisible by", N_BATCHES)
TIMESTEPS_PER_BATCH = N_TIMESTEPS_PER_KERNEL*GEN_KERNEL_SIZE // N_BATCHES
if N_TIMESTEPS_PER_KERNEL*GEN_KERNEL_SIZE % N_BATCHES != 0:
    raise ValueError("Could not calculate SAMPLES_PER_BATCH: Total kernel samples not divisible by", N_BATCHES)

print("v-"*39 + "v")
print("Total # of input samples:", TOTAL_SAMPLES_IN)
print("Timesteps per kernel:", N_TIMESTEPS_PER_KERNEL)
print("Batches per layer:", N_BATCHES)
print("Samples per batch:", SAMPLES_PER_BATCH)
print("Timesteps per batch:", TIMESTEPS_PER_BATCH)
print("Number of channels:", N_CHANNELS)
# if GEN_MODE == 1:
#     print("Will create", N_RNN_LAYERS, "layers of", N_UNITS, "units.")
# elif GEN_MODE == 0:
#     print("Will create 1 Pre-Dense layer of", SAMPLES_PER_BATCH, "filters.")    
#     if N_PRE_DENSE_LAYERS > 1:
#         print("Will create", N_PRE_DENSE_LAYERS-1, "Pre-Dense layers of", N_UNITS*N_TIMESTEPS, "filters.")
#     print("Will create", N_DENSE_LAYERS-1, "Dense layers of", SAMPLES_PER_BATCH, "units.")
#     print("Will create 1 Dense layer of", SAMPLES_PER_BATCH, "units.")
#     if N_POST_DENSE_LAYERS > 1:
#         for i in range(1,N_POST_DENSE_LAYERS):
#                 n_filts = i*N_POST_DENSE_FILTERS
#                 print("Will create 1 Post-Dense layer of", n_filts, "filters.")
#     else:
#         print("Will create 1 Post-Dense layer of", N_UNITS*N_TIMESTEPS//20, "filters.")
output_samples = N_BATCHES*N_CHANNELS*N_TIMESTEPS_PER_KERNEL*GEN_KERNEL_SIZE
print("Total # of output samples:", output_samples)
print("^-"*39 + "^")

def v_print(*s):
    if VERBOSITY_LEVEL >= 1:
        print(*s)
def vv_print(*s):
    if VERBOSITY_LEVEL >= 2:
        print(*s)