class Config:
    # Data paths
    DATA_DIR = r'C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset'
    DATA_JSON = r"C:\Users\Adib\PycharmProjects\Hippocampus_Segmentation\dataset\dataset.json"

    # Training Config
    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    # Data preprocessing settings
    RESIZE_WIDTH = 256
    RESIZE_HEIGHT = 256
    NORMALIZE_MEAN = 0.5
    NORMALIZE_STD = 0.5

    # GPU configuration
    USE_GPU = True
    USE_GPU_WITH_MORE_MEMORY = True
    USE_GPU_WITH_MORE_COMPUTE_CAPABILITY = False
