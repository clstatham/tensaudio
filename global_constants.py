import numpy as np

RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "examples_speech"
EXAMPLE_RESULTS_DIR = "example_results"
INPUTS_DIR = "inputs_speech"

PLOTS_DIR = "D:\\tensaudio_plots"
TRAINING_DIR = "D:\\tensaudio_training"

MODEL_DIR = "D:\\tensaudio_models"

SAVE_EVERY_ITERS = 250*10 # approx. every 30 minutes
SAVE_MODEL_EVERY_ITERS = 250*60 # approx. every hour

VERBOSE_OUTPUT = True

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!
GEN_MODE = 0 # 0 = convolution/dense/deconvolution mode, 1 = RNN mode
USE_REAL_AUDIO = True
SAMPLE_RATE = 24000
SUBTYPE = 'PCM_16'
SECONDS_OF_AUDIO = 4
SLICE_START = 0
N_RNN_LAYERS = 4
N_CONV_LAYERS = 2
N_DENSE_LAYERS = 8
N_DECONV_LAYERS = 1
N_TIMESTEPS = 50
KERNEL_SIZE = 16
GENERATOR_LR = 0.5

# If you change ANY of the following values, you MUST empty
# MODEL_DIR/dis_ckpts folder or the discsriminator model will give
# an error!
N_DIS_LAYERS = 14
DISCRIMINATOR_LR = 0.5

# DO NOT CHANGE THESE
TARGET_LEN_OVERRIDE = int(SAMPLE_RATE * SECONDS_OF_AUDIO) - SLICE_START
N_BATCHES = TARGET_LEN_OVERRIDE // (KERNEL_SIZE * N_TIMESTEPS * 2)
if TARGET_LEN_OVERRIDE % (KERNEL_SIZE * N_TIMESTEPS * 2) != 0:
    raise ValueError("Could not calculate N_BATCHES: Total length of audio not divisible by", (KERNEL_SIZE * N_TIMESTEPS * 2))
SAMPLES_PER_BATCH = TARGET_LEN_OVERRIDE // N_BATCHES
if TARGET_LEN_OVERRIDE % N_BATCHES != 0:
    raise ValueError("Could not calculate SAMPLES_PER_BATCH: Total length of audio not divisible by", (KERNEL_SIZE * N_TIMESTEPS * 2))
N_UNITS = TARGET_LEN_OVERRIDE // (N_TIMESTEPS * N_BATCHES)
if N_UNITS <= KERNEL_SIZE:
    raise ValueError("N_UNITS must be greater than KERNEL_SIZE - pick a smaller N_TIMESTEPS!", N_UNITS)
N_DECONV_BATCHES = N_BATCHES
N_DECONV_FILTERS = TARGET_LEN_OVERRIDE // (2*(N_DECONV_LAYERS+1))
if TARGET_LEN_OVERRIDE % (2*(N_DECONV_LAYERS+1)) != 0:
    raise ValueError("Could not calculate N_DECONV_FILTERS: Total length of audio not divisible by", 2*(N_DECONV_LAYERS+1))

print("v-"*39 + "v")
print("Total # of input samples:", TARGET_LEN_OVERRIDE)
print("Timesteps per layer:", N_TIMESTEPS)
print("Batches per layer:", N_BATCHES)
print("Samples per batch:", SAMPLES_PER_BATCH)
if GEN_MODE == 1:
    print("Will create", N_RNN_LAYERS, "layers of", N_UNITS, "units.")
elif GEN_MODE == 0:
    print("Will create 1 convolution layer of", SAMPLES_PER_BATCH, "filters.")
    if N_CONV_LAYERS > 1:
        print("Will create", N_CONV_LAYERS-1, "convolution layers of", N_UNITS*N_TIMESTEPS, "filters.")
    print("Will create", N_DENSE_LAYERS-1, "dense layers of", N_TIMESTEPS, "units.")
    print("Will create 1 dense layer of", SAMPLES_PER_BATCH, "units.")
    if N_DECONV_LAYERS > 1:
        for i in range(1,N_DECONV_LAYERS):
                n_filts = i*N_DECONV_FILTERS
                print("Will create 1 deconvolution layer of", n_filts, "filters.")
    else:
        print("Will create 1 deconvolution layer of", N_UNITS*N_TIMESTEPS//20, "filters.")
output_samples = N_BATCHES*(N_UNITS*N_TIMESTEPS)
print("Total # of output samples:", output_samples)
print("^-"*39 + "^")

def v_print(*s):
    if VERBOSE_OUTPUT:
        print(*s)