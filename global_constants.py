RESOURCES_DIR = "D:\\tensaudio_resources"
EXAMPLES_DIR = "examples_speech"
EXAMPLE_RESULTS_DIR = "example_results"
INPUTS_DIR = "inputs_speech"

MODEL_DIR = "D:\\tensaudio_models"

SAVE_EVERY_ITERS = 250
SAVE_MODEL_EVERY_ITERS = 250

VERBOSE_OUTPUT = False
def v_print(*s):
    if VERBOSE_OUTPUT:
        print(*s)

SAMPLE_RATE = 8000
#TARGET_SR = 44100
#TARGET_SR = 11025
SUBTYPE = 'PCM_16'
N_FFT = 2048
N_LAYERS = 16
N_TIMESTEPS = 50
N_BATCHES = 4
GENERATOR_LR = 0.05
DISCRIMINATOR_LR = 0.2
SLICE_START = 0
SECONDS_OF_AUDIO = 2
TARGET_LEN_OVERRIDE = SAMPLE_RATE * SECONDS_OF_AUDIO
if TARGET_LEN_OVERRIDE % N_TIMESTEPS != 0:
    raise ValueError("Total length of audio not divisible by", N_TIMESTEPS)
N_UNITS = TARGET_LEN_OVERRIDE // N_TIMESTEPS
print("Will create", N_LAYERS, "layers of", N_UNITS, "units.")