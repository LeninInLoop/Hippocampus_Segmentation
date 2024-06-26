class Config:
    # Data paths
    DATA_DIR = 'data/dataset'
    TRAIN_DIR = f'{DATA_DIR}/train'
    VALIDATION_DIR = f'{DATA_DIR}/validation'
    TEST_DIR = f'{DATA_DIR}/test'

    # Model hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    OPTIMIZER = 'adam'
    WEIGHT_DECAY = 0.0001

    # Model architecture settings
    NUM_CLASSES = 2
    ENCODER_DEPTH = 5
    DECODER_DEPTH = 5
    FILTERS_START = 64
    FILTERS_END = 64

    # Data preprocessing settings
    RESIZE_WIDTH = 256
    RESIZE_HEIGHT = 256
    NORMALIZE_MEAN = 0.5
    NORMALIZE_STD = 0.5

    # Training and evaluation settings
    EARLY_STOPPING_PATIENCE = 5
    LOGGING_FREQUENCY = 10
    CHECKPOINT_FREQUENCY = 10

    # GPU configuration
    USE_GPU = True
    USE_GPU_WITH_MORE_MEMORY = True
    USE_GPU_WITH_MORE_COMPUTE_CAPABILITY = False
