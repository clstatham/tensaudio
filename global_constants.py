RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "examples_speech"
EXAMPLE_RESULTS_DIR = "example_results"
INPUTS_DIR = "inputs_speech"

MODEL_DIR = "D:\\tensaudio_models"

SAVE_EVERY_ITERS = 250
SAVE_MODEL_EVERY_ITERS = 250

VERBOSE_OUTPUT = False


# If you change ANY of the following values, you MUST empty
# MODEL_DIR/gen_ckpts folder or the generator model will give
# an error!

GEN_MODE = 0 # 0 = dense/deconvolution mode, 1 = RNN mode
SAMPLE_RATE = 8000
#TARGET_SR = 44100
#TARGET_SR = 11025
SUBTYPE = 'PCM_16'
N_LAYERS = 16
N_DENSE_LAYERS = 16
N_DECONV_LAYERS = 4
N_BATCHES = 4
GENERATOR_LR = 0.05
DISCRIMINATOR_LR = 0.2
SLICE_START = 0
SECONDS_OF_AUDIO = 2
TARGET_LEN_OVERRIDE = SAMPLE_RATE * SECONDS_OF_AUDIO
N_TIMESTEPS = 4
N_UNITS = TARGET_LEN_OVERRIDE // N_TIMESTEPS


if TARGET_LEN_OVERRIDE % N_TIMESTEPS != 0:
    raise ValueError("Total length of audio not divisible by", N_TIMESTEPS)
print("Will create", N_LAYERS, "layers of", N_UNITS, "units.")

def v_print(*s):
    if VERBOSE_OUTPUT:
        print(*s)