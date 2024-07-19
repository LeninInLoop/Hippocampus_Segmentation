class Config:
    # Data paths
    VANDERBILT_DATA_DIR = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset'
    VANDERBILT_DATA_JSON = VANDERBILT_DATA_DIR + r"\dataset.json"

    # Model
    SLOPE_OF_LEAKY_RELU = 0.01

    # Training settings
    TRAIN_RATIO = 0.6,  # 60% for training
    VAL_RATIO = 0.1,  # 10% for validation
    TEST_RATIO = 0.3,  # 30% for testing
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    NUM_EPOCHS = 20
    VAL_EPOCHS = 4
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
    BEST_MODEL_SAVE_PATH = r".\Output\best_model.pth"
    LOGS_FOLDER = r".\Output\logs"
    TEST_LOGS_FOLDER = LOGS_FOLDER + r"\Test"
