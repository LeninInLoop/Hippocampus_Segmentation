class Config:
    # Data paths
    DATA_DIR = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset'
    DATA_JSON = DATA_DIR + r"\dataset.json"

    # Training Config
    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # Data preprocessing settings
    PADDING_TARGET_SHAPE = (64, 64, 64)

    # GPU configuration
    USE_GPU = True
    USE_GPU_WITH_MORE_MEMORY = True
    USE_GPU_WITH_MORE_COMPUTE_CAPABILITY = False
