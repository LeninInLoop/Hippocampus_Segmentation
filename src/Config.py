class Config:
    # Data paths
    DATA_DIR = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset'
    DATA_JSON = DATA_DIR + r"\dataset.json"

    # Training settings
    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    NUM_EPOCHS = 20
    VAL_EPOCHS = 4
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 1e-5

    # Optimizer settings
    OPTIMIZER = 'Adam'  # 'Adam' or 'SGD'
    MOMENTUM = 0.9  # Only used if OPTIMIZER is 'SGD'

    # Learning rate scheduler settings
    LR_SCHEDULER = 'ReduceLROnPlateau'  # 'StepLR' or 'ReduceLROnPlateau'
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1
    LR_PATIENCE = 5  # Only used if LR_SCHEDULER is 'ReduceLROnPlateau'

    # Data preprocessing settings
    PADDING_TARGET_SHAPE = (64, 64, 64)

    # GPU configuration
    USE_GPU = True
    USE_GPU_WITH_MORE_MEMORY = True
    USE_GPU_WITH_MORE_COMPUTE_CAPABILITY = False

    # Model saving
    MODEL_SAVE_PATH = DATA_DIR + r"\Output\best_unet3d_model.pth"
    LOGS_FOLDER = r"Output\logs"
