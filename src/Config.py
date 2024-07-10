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
    LEARNING_RATE = 0.005

    # Optimizer settings
    OPTIMIZER = 'Adam'

    # Data preprocessing settings
    PADDING_TARGET_SHAPE = (48, 64, 48)

    # GPU configuration
    USE_GPU = True
    USE_GPU_WITH_MORE_MEMORY = True
    USE_GPU_WITH_MORE_COMPUTE_CAPABILITY = False

    # Model saving
    MODEL_SAVE_PATH = DATA_DIR + r"\Output\best_unet3d_model.pth"
    LOGS_FOLDER = DATA_DIR + r"\Output\logs"
