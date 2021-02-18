RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "kicks"
EXAMPLE_RESULTS_DIR = "fire"
INPUTS_DIR = "inputs_kicks"

PLOTS_DIR = "D:\\tensaudio_plots"
TRAINING_DIR = "D:\\tensaudio_training"

MODEL_DIR = "D:\\tensaudio_models"

# set to 0 to disable periodically generating progress updates
SAVE_EVERY_ITERS = 1000
# set to 0 to disable periodically saving model
SAVE_MODEL_EVERY_ITERS = 1000*10

VERBOSE_OUTPUT = False

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!
INPUT_MODE = 'direct'   # 'direct' = direct comparison of example and example result
                        # 'conv' = comparison of example and convolved example result

GEN_MODE = 2            # 0 = RNN/Hilbert mode
                        # 1 = RNN/Audio mode
                        # 2 = Conv/Hilbert mode
                        # 3 = Conv/Audio mode
USE_REAL_AUDIO = False
SAMPLE_RATE = 16000
SUBTYPE = 'PCM_16'
SECONDS_OF_AUDIO = 4
SLICE_START = 0
N_RNN_LAYERS = 4
N_PRE_DENSE_LAYERS = 1
N_DENSE_LAYERS = 1
N_POST_DENSE_LAYERS = 1
N_TIMESTEPS = 50
KERNEL_SIZE = 8
GENERATOR_LR = 0.001

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/dis_ckpts folder or the discsriminator model will give
# an error!
N_DIS_LAYERS = 14
DISCRIMINATOR_LR = 0.5

# DO NOT CHANGE ANYTHING BELOW THIS LINE!
# ---------------------------------------------------------------------------------

import numpy as np

TOTAL_SAMPLES = int(SAMPLE_RATE * SECONDS_OF_AUDIO) - SLICE_START
N_BATCHES = TOTAL_SAMPLES // (KERNEL_SIZE * N_TIMESTEPS * 2)
if TOTAL_SAMPLES % (KERNEL_SIZE * N_TIMESTEPS * 2) != 0:
    raise ValueError("Could not calculate N_BATCHES: Total length of audio not divisible by", (KERNEL_SIZE * N_TIMESTEPS * 2))
SAMPLES_PER_BATCH = TOTAL_SAMPLES // N_BATCHES
if TOTAL_SAMPLES % N_BATCHES != 0:
    raise ValueError("Could not calculate SAMPLES_PER_BATCH: Total length of audio not divisible by", (KERNEL_SIZE * N_TIMESTEPS * 2))
N_UNITS = TOTAL_SAMPLES // (N_TIMESTEPS * N_BATCHES)
if N_UNITS <= KERNEL_SIZE:
    raise ValueError("N_UNITS must be greater than KERNEL_SIZE - pick a smaller N_TIMESTEPS!", N_UNITS)
if TOTAL_SAMPLES % (N_TIMESTEPS * N_BATCHES) != 0:
    raise ValueError("Could not calculate N_UNITS: Total length of audio not divisible by", (N_TIMESTEPS * N_BATCHES))
N_POST_DENSE_BATCHES = N_BATCHES
N_PRE_DENSE_FILTERS = SAMPLES_PER_BATCH // (2*(N_PRE_DENSE_LAYERS+1))
if SAMPLES_PER_BATCH % (2*(N_PRE_DENSE_LAYERS+1)) != 0:
    raise ValueError("Could not calculate N_POST_DENSE_FILTERS: Samples per batch not divisible by", 2*(N_PRE_DENSE_LAYERS+1))
N_POST_DENSE_FILTERS = TOTAL_SAMPLES // (2*(N_POST_DENSE_LAYERS+1))
if TOTAL_SAMPLES % (2*(N_POST_DENSE_LAYERS+1)) != 0:
    raise ValueError("Could not calculate N_POST_DENSE_FILTERS: Total length of audio not divisible by", 2*(N_POST_DENSE_LAYERS+1))

print("v-"*39 + "v")
print("Total # of input samples:", TOTAL_SAMPLES)
print("Timesteps per layer:", N_TIMESTEPS)
print("Batches per layer:", N_BATCHES)
print("Samples per batch:", SAMPLES_PER_BATCH)
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
output_samples = N_BATCHES*(N_UNITS*N_TIMESTEPS)
print("Total # of output samples:", output_samples)
print("^-"*39 + "^")

def v_print(*s):
    if VERBOSE_OUTPUT:
        print(*s)