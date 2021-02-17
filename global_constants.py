import numpy as np

RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "examples_speech"
EXAMPLE_RESULTS_DIR = "example_results"
INPUTS_DIR = "inputs_speech"

MODEL_DIR = "D:\\tensaudio_models"

SAVE_EVERY_ITERS = 250*10 # approx. every 30 minutes
SAVE_MODEL_EVERY_ITERS = 250*60 # approx. every hour

VERBOSE_OUTPUT = False


# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!

GEN_MODE = 0 # 0 = convolution/dense/deconvolution mode, 1 = RNN mode
SAMPLE_RATE = 24000
#SAMPLE_RATE = 11025
SECONDS_OF_AUDIO = 4
SUBTYPE = 'PCM_16'
N_RNN_LAYERS = 4
N_CONV_LAYERS = 2
N_DENSE_LAYERS = 8
N_DECONV_LAYERS = 1
N_TIMESTEPS = 50
GENERATOR_LR = 0.01
DISCRIMINATOR_LR = 0.02
SLICE_START = 0
KERNEL_SIZE = 16

# DO NOT CHANGE THESE
TARGET_LEN_OVERRIDE = int(SAMPLE_RATE * SECONDS_OF_AUDIO)
N_BATCHES = TARGET_LEN_OVERRIDE // (KERNEL_SIZE * N_TIMESTEPS * 2)
N_UNITS = TARGET_LEN_OVERRIDE // (N_TIMESTEPS * N_BATCHES)
if N_UNITS <= KERNEL_SIZE:
    raise ValueError("N_UNITS must be greater than KERNEL_SIZE - pick a smaller N_TIMESTEPS!", N_UNITS)
N_DECONV_BATCHES = N_BATCHES
if TARGET_LEN_OVERRIDE % (N_TIMESTEPS * N_BATCHES) != 0:
    raise ValueError("Total length of audio not divisible by", (N_TIMESTEPS * N_BATCHES))


print("v-"*39 + "v")
print("Timesteps per layer:", N_TIMESTEPS)
print("Batches per layer:", N_BATCHES)
if GEN_MODE == 1:
    print("Will create", N_RNN_LAYERS, "layers of", N_UNITS, "units.")
elif GEN_MODE == 0:
    print("Will create", N_DENSE_LAYERS, "dense layers of", N_UNITS, "units.")
    print("Will create", N_DECONV_LAYERS, "deconvolution layers of", N_UNITS, "units.")
print("^-"*39 + "^")

def v_print(*s):
    if VERBOSE_OUTPUT:
        print(*s)