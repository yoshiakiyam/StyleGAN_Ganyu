import torch



DATASET                 = "data"
START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE           = 0.001
                          # 4    8   16     32  64  128
BATCH_SIZES             = [0, 64, 64,  64, 64, 64]
CHANNELS_IMG            = 3
Z_DIM                   = 128
W_DIM                   = 128
IN_CHANNELS             = 256
LAMBDA_GP               = 10
PROGRESSIVE_EPOCHS      = [200] * len(BATCH_SIZES)