#-----------------------------------------------
#Config that does not have impact on performance
#-----------------------------------------------

RANDOM_SEED = 0b011011


#-----------------------------------------------
#Training
#-----------------------------------------------

EPOCHS_SAVE_START = 0
#Path of output of validation. 
OUTPUT_DIR_PATH = "/media/youngwon/Neo/NeoChoi/Projects/AECNN_for_Speech_Enhancement/test"

LOGGER_PATH = "/media/youngwon/Neo/NeoChoi/Projects/AECNN_for_Speech_Enhancement/tb_logger"

MAX_EPOCHS= 50

#-----------------------------------------------
#1. Dataset
#-----------------------------------------------
#directory that have every dataset in it.
DATA_DIR = "/media/youngwon/Neo/NeoChoi/TIL/TIL_Dataset/AECNN_enhancement"

#Noisy dataset
NOISY_TRAIN = "TIMIT_decoded/TRAIN"
NOISY_TEST = "TIMIT_decoded/TEST"

#Target dataset
TARGET_TRAIN = "TIMIT/TRAIN"
TARGET_TEST = "TIMIT/TEST"

BATCH_SIZE = 8
SEG_LEN = 2

 
#-----------------------------------------------
#2. Model
#-----------------------------------------------
WINDOW_SIZE = 2048
HOP_SIZE = 512

NUM_LAYERS = 8
KERNEL_SIZE = 11

#-----------------------------------------------
#3. Loss
#-----------------------------------------------
LOSS_TYPE = "STFTLoss"


#other losses...
#LOSS_TYPE = "SISNRLoss"
# LOSS_TYPE = "MelSpecLoss"
#LOSS_TYPE = "DemucsLoss"

#for STFT Loss
STFTLOSS_WINDOW_SIZE = 512
STFTLOSS_HOP_SIZE = 256

#-----------------------------------------------
#4. Optimizer
#-----------------------------------------------
INITIAL_LR = 0.001
LR_GAMMA = 0.9
