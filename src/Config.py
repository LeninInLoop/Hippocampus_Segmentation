class Config:
    # Data paths
    VANDERBILT_DATA_DIR = r'/content/Hippocampus_Segmentation/dataset/Task04_Hippocampus'
    VANDERBILT_DATA_JSON = VANDERBILT_DATA_DIR + r"./dataset.json"

    # Training settings
    TRAIN_AND_VAL_SPLIT_RATIO = 0.8
    TEST_SPLIT_RADIO = 0.8
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    NUM_EPOCHS = 20
    VAL_EPOCHS = 4
    LEARNING_RATE = 0.0001

    # Optimizer settings
    OPTIMIZER = 'Adam'

    # Data preprocessing settings
    PADDING_TARGET_SHAPE = (48, 64, 48)

    # GPU configuration
    USE_GPU = True
    USE_GPU_WITH_MORE_MEMORY = True
    USE_GPU_WITH_MORE_COMPUTE_CAPABILITY = False

    # Model saving
    BEST_MODEL_SAVE_PATH = r"./Output/best_unet3d_model.pth"
    LOGS_FOLDER = r"./Output/logs"
    TEST_LOGS_FOLDER = LOGS_FOLDER + r"/Test"
