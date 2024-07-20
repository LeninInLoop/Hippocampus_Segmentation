from src.Config import Config
import json


def print_vanderbilt_dataset_info(data_json=Config.VANDERBILT_DATA_JSON):
    """
    Print formatted information about the dataset.

    Args:
        data_json (dict): A dictionary containing dataset information.
    """

    # Load the JSON file
    try:
        with open(data_json, 'r') as file:
            data_json = json.load(file)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {data_json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {data_json}")
        return

    print("\n" + "=" * 50)
    print("Dataset Information".center(50))
    print("=" * 50)

    info_items = [
        ("Dataset Name", data_json['name']),
        ("Description", data_json['description']),
        ("Reference", data_json['reference']),
        ("Licence", data_json['licence']),
        ("Release Version", data_json['relase']),
        ("Tensor Image Size", data_json['tensorImageSize']),
        ("Modality", data_json['modality']['0']),
        ("Number of Training Images", data_json['numTraining']),
        ("Number of Test Images", data_json['numTest'])
    ]

    for label, value in info_items:
        print(f"{label:<25} {value}")

    print("\n" + "=" * 50 + "\n")


def print_cuda_device_info(cuda_devices):
    """
    Print formatted information about CUDA devices.

    Args:
        cuda_devices (list): A list of dictionaries containing CUDA device information.
    """
    print("\n" + "=" * 50)
    print("CUDA Devices Information".center(50))
    print("=" * 50)

    for index, device in enumerate(cuda_devices, 1):
        print(f"\nDevice {index}/{len(cuda_devices)}:")
        print("-" * 50)
        device_info = [
            ("Device ID", device['device_id']),
            ("Device Name", device['name']),
            ("Device Memory", device['memory']),
            ("Device Compute Capability", device['compute_capability'])
        ]
        for label, value in device_info:
            print(f"{label:<30} {value}")

    print("\n" + "=" * 50 + "\n")


def print_batch_info(loader, loader_name):
    """Print sample batch shapes for a given loader."""
    print("\n" + "=" * 50)
    print(f"{loader_name} Batch Information".center(50))
    print("=" * 50)

    for batch_images, batch_labels in loader:
        print(f"Batch image shape: {batch_images.shape}")
        print(f"Batch label shape: {batch_labels.shape}")
        break
    print(f"Number of samples: {len(loader)}")

    print("\n" + "=" * 50 + "\n")


def print_config():
    """Print the configuration settings in a more readable format."""
    print("\n" + "=" * 50)
    print("Configuration Settings".center(50))
    print("=" * 50)

    config_items = [
        ("Data Paths", [
            ("DATA_DIR", Config.VANDERBILT_DATA_DIR),
            ("DATA_JSON", Config.VANDERBILT_DATA_JSON)
        ]),
        ("Training Settings", [
            ("USE_KFOLD", Config.USE_KFOLD),
            ("NUM_OF_FOLDS", Config.NUM_OF_FOLDS),
            ("TRAIN_SPLIT_RATIO", Config.TRAIN_RATIO),
            ("VAL_SPLIT_RATIO", Config.VAL_RATIO),
            ("TEST_SPLIT_RATIO", Config.TEST_RATIO),
            ("BATCH_SIZE", Config.BATCH_SIZE),
            ("NUM_WORKERS", Config.NUM_WORKERS),
            ("NUM_EPOCHS", Config.NUM_EPOCHS),
            ("VAL_EPOCHS", Config.VAL_EPOCHS),
            ("LEARNING_RATE", Config.LEARNING_RATE)
        ]),
        ("Optimizer", [
            ("OPTIMIZER", Config.OPTIMIZER)
        ]),
        ("Data Preprocessing", [
            ("PADDING_TARGET_SHAPE", Config.PADDING_TARGET_SHAPE)
        ]),
        ("GPU Configuration", [
            ("USE_GPU", Config.USE_GPU),
            ("USE_GPU_WITH_MORE_MEMORY", Config.USE_GPU_WITH_MORE_MEMORY),
            ("USE_GPU_WITH_MORE_COMPUTE_CAPABILITY", Config.USE_GPU_WITH_MORE_COMPUTE_CAPABILITY)
        ]),
        ("Model Saving", [
            ("MODEL_SAVE_PATH", Config.BEST_MODEL_SAVE_PATH),
            ("LOGS_FOLDER", Config.LOGS_FOLDER)
        ])
    ]

    for section, items in config_items:
        print(f"\n{section}:")
        print("-" * 50)
        for key, value in items:
            print(f"{key:<35} {value}")

    print("\n" + "=" * 50 + "\n")
