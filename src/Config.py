class Config:
    # Data paths
    VANDERBILT_DATA_DIR = r'./dataset/Task04_Hippocampus'
    VANDERBILT_DATA_JSON = VANDERBILT_DATA_DIR + r"/dataset.json"

    # Model
    SLOPE_OF_LEAKY_RELU = 0.01

    # Training settings
    USE_KFOLD = True
    NUM_OF_FOLDS = 5
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    NUM_EPOCHS = 40
    VAL_EPOCHS = 40
    LEARNING_RATE = 0.01

    # Optimizer settings
    OPTIMIZER = 'Adam'

    # Data preprocessing settings
    PADDING_TARGET_SHAPE = (48, 64, 48)

    # GPU configuration
    USE_GPU = True
    USE_GPU_WITH_MORE_MEMORY = True
    USE_GPU_WITH_MORE_COMPUTE_CAPABILITY = False

    # Model saving
    BEST_MODEL_SAVE_PATH = r"./Output/best_model.pth"
    LOGS_FOLDER = r"./Output/logs"
    TEST_LOGS_FOLDER = LOGS_FOLDER + r"/Test"
