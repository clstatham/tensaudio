RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "kicks"
EXAMPLE_RESULTS_DIR = "hihats"
INPUTS_DIR = "inputs_kicks"

PLOTS_DIR = "D:\\tensaudio_plots"
TRAINING_DIR = "D:\\tensaudio_training"

MODEL_DIR = "D:\\tensaudio_models"

# set to 0 to run until Ctrl+C is input in the terminal
MAX_ITERS = 100

# set to 0 to disable periodically generating progress updates
SAVE_EVERY_ITERS = 10
# set to 0 to disable periodically saving model
SAVE_MODEL_EVERY_ITERS = 10000*10

VERBOSE_OUTPUT = True

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!
INPUT_MODE = 'direct'   # 'direct' = direct comparison of example and example result
                        # 'conv' = comparison of example and convolved example result

GEN_MODE = 3            # 0 = RNN/Hilbert mode
                        # 1 = RNN/Audio mode
                        # 2 = Conv/Hilbert mode
                        # 3 = Conv/Audio mode
USE_REAL_AUDIO = True
SAMPLE_RATE = 24000
SUBTYPE = 'PCM_16'
SECONDS_OF_AUDIO = 2.4
SLICE_START = 0
N_RNN_LAYERS = 4
N_PREPROCESS_LAYERS = 2
N_PROCESS_LAYERS = 128
N_POSTPROCESS_LAYERS = 6 # THIS INCREASES MEMORY USE **EXPONENTIALLY**
N_TIMESTEPS_PER_KERNEL = 300
BATCH_OPTIMIZATION_FACTOR = 1 # DO NOT TOUCH (for now)
KERNEL_SIZE = 4
GENERATOR_LR = 0.001
GENERATOR_MOMENTUM = 0.1

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/dis_ckpts folder or the discsriminator model will give
# an error!
N_DIS_LAYERS = 14
DISCRIMINATOR_LR = 0.005
DISCRIMINATOR_MOMENTUM = 0.1

# DO NOT CHANGE ANYTHING BELOW THIS LINE!
# ---------------------------------------------------------------------------------

import numpy as np

TOTAL_SAMPLES = int(SAMPLE_RATE * SECONDS_OF_AUDIO) - SLICE_START
if GEN_MODE in [0, 2]:
    N_CHANNELS = 2
else:
    N_CHANNELS = 1
N_BATCHES = BATCH_OPTIMIZATION_FACTOR * TOTAL_SAMPLES // (N_CHANNELS * KERNEL_SIZE * N_TIMESTEPS_PER_KERNEL)
if TOTAL_SAMPLES % (KERNEL_SIZE * N_TIMESTEPS_PER_KERNEL * 2) != 0:
    raise ValueError("Could not calculate N_BATCHES: Total length of audio not divisible by", (KERNEL_SIZE * N_TIMESTEPS_PER_KERNEL * 2))
SAMPLES_PER_BATCH = TOTAL_SAMPLES // N_BATCHES
if TOTAL_SAMPLES % N_BATCHES != 0:
    raise ValueError("Could not calculate SAMPLES_PER_BATCH: Total length of audio not divisible by", (KERNEL_SIZE * N_TIMESTEPS_PER_KERNEL * 2))
N_POST_DENSE_BATCHES = N_BATCHES
N_PRE_DENSE_FILTERS = SAMPLES_PER_BATCH // (2*(N_PREPROCESS_LAYERS+1))
if SAMPLES_PER_BATCH % (2*(N_PREPROCESS_LAYERS+1)) != 0:
    raise ValueError("Could not calculate N_POST_DENSE_FILTERS: Samples per batch not divisible by", 2*(N_PREPROCESS_LAYERS+1))
N_POST_DENSE_FILTERS = TOTAL_SAMPLES // N_BATCHES
if TOTAL_SAMPLES % (N_BATCHES) != 0:
    raise ValueError("Could not calculate N_POST_DENSE_FILTERS: Total length of audio not divisible by", (N_BATCHES))

print("v-"*39 + "v")
print("Total # of input samples:", TOTAL_SAMPLES)
print("Timesteps per kernel:", N_TIMESTEPS_PER_KERNEL)
print("Batches per layer:", N_BATCHES)
print("Samples per batch:", SAMPLES_PER_BATCH)
print("Number of channels:", N_CHANNELS)
print("Final layer output size:", N_POST_DENSE_FILTERS)
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
output_samples = N_BATCHES*N_CHANNELS*N_TIMESTEPS_PER_KERNEL*KERNEL_SIZE
print("Total # of output samples:", output_samples)
print("^-"*39 + "^")

def v_print(*s):
    if VERBOSE_OUTPUT:
        print(*s)