import os.path
from src.train import *
from src.data import *
from src.models import UNet3D


def setup_gpu():
    """Set up and configure GPU if available."""
    if not SystemInfo.is_cuda_available():
        print("No GPU is Available")
        return None

    print("CUDA is available.")
    cuda_devices = SystemInfo.get_cuda_devices()
    print_cuda_device_info(cuda_devices)

    if Config.USE_GPU_WITH_MORE_MEMORY:
        System.set_cuda_device_with_highest_mem(cuda_devices)
    elif Config.USE_GPU_WITH_MORE_COMPUTE_CAPABILITY:
        System.set_cuda_device_with_highest_compute(cuda_devices)

    return torch.device("cuda")


def initialize_model(device):
    """Initialize and return the model."""
    model = UNet3D(in_channels=1, out_channels=3, feat_channels=32).to(device)
    return model


def get_optimizer(model):
    """Get the optimizer based on configuration."""
    if Config.OPTIMIZER == 'Adam':
        return Adam(model.parameters(), lr=Config.LEARNING_RATE)
    elif Config.OPTIMIZER == 'AdamW':
        return AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    elif Config.OPTIMIZER == 'RMSprop':
        return RMSprop(model.parameters(), lr=Config.LEARNING_RATE)
    # Add more optimizer options here if needed
    raise ValueError(f"Unsupported optimizer: {Config.OPTIMIZER}")


def print_fold_results(fold_num, results, is_test=False):
    result_type = "Test" if is_test else "Cross-Validation"
    print(f"\nFold {fold_num} {result_type} Results:")
    print("-" * 40)

    mean_dice_1, mean_dice_2 = results['mean_multi_dice']
    std_dice_1, std_dice_2 = results['std_multi_dice']

    print(f"Class 1: {mean_dice_1:.4f} +/- {std_dice_1:.4f}")
    print(f"Class 2: {mean_dice_2:.4f} +/- {std_dice_2:.4f}")

    mean_multi_dice = (mean_dice_1 + mean_dice_2) / 2
    std_multi_dice = (std_dice_1 + std_dice_2) / 2

    print(f"Mean Multi-Dice: {mean_multi_dice:.4f} +/- {std_multi_dice:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")


def print_average_results(results):
    print("\nAverage Test Results Across All Folds:")
    print("=" * 40)

    mean_dice_1, mean_dice_2 = results['mean_multi_dice']
    std_dice_1, std_dice_2 = results['std_multi_dice']

    print(f"Class 1: {mean_dice_1:.4f} +/- {std_dice_1:.4f}")
    print(f"Class 2: {mean_dice_2:.4f} +/- {std_dice_2:.4f}")

    mean_multi_dice = (mean_dice_1 + mean_dice_2) / 2
    std_multi_dice = (std_dice_1 + std_dice_2) / 2

    print(f"Mean Multi-Dice: {mean_multi_dice:.4f} +/- {std_multi_dice:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Overall Mean Dice: {results['overall_mean_dice']:.4f}")
    print(f"Overall Std Dice: {results['overall_std_dice']:.4f}")


def main():
    # Setup device
    if not Config.USE_GPU:
        print("GPU usage is disabled in config. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = setup_gpu()
        if device is None:
            return

    # Print configuration and dataset info
    print_config()

    if not os.path.isdir(Config.VANDERBILT_DATA_DIR):
        download_dataset(Config.VANDERBILT_DATA_DIR)

    print_vanderbilt_dataset_info()

    # Initialize model and optimizer
    model = initialize_model(device)
    optimizer = get_optimizer(model)

    # Setup data loader strategy
    hippocampus_strategy = VanderbiltHippocampusDatasetStrategy()

    if Config.USE_KFOLD:
        data_loader_factory = KFoldDataLoaderFactory(
            dataset_strategy=hippocampus_strategy,
            train_transform=train_transform,
            val_transform=validation_transform,
            test_transform=validation_transform,
            k_folds=Config.NUM_OF_FOLDS,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            test_ratio=Config.TEST_RATIO
        )
        training_strategy = KFoldTrainingStrategy(data_loader_factory)
    else:
        data_loader_factory = DefaultDataLoaderFactory(
            dataset_strategy=hippocampus_strategy,
            train_transform=train_transform,
            val_transform=validation_transform,
            test_transform=validation_transform,
            train_ratio=Config.TRAIN_RATIO,
            val_ratio=Config.VAL_RATIO,
            test_ratio=Config.TEST_RATIO,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS
        )
        training_strategy = StandardTrainingStrategy(data_loader_factory)

    # Execute training
    val_results, final_results = training_strategy.execute(model, device, optimizer)

    # Process results
    if Config.USE_KFOLD:
        print("\nK-fold Cross-Validation Results:")
        print("================================")
        for i, result in enumerate(val_results):
            print_fold_results(i + 1, result)

        print("\nFinal Test Results:")
        print("===================")
        for i, result in enumerate(final_results[0]):
            print_fold_results(i + 1, result, is_test=True)

        print_average_results(final_results[1])
    else:
        print("\nValidation Results:")
        print("===================")
        print_fold_results(1, val_results)

        print("\nFinal Test Results:")
        print("===================")
        print_fold_results(1, final_results, is_test=True)

    # Uncomment the following lines if you want to generate a model diagram
    # visualizer = UNet3DVisualizer(model)
    # visualizer.generate_diagram((16, 1, 48, 64, 48), filename="ModelDiagram", format="png")


if __name__ == '__main__':
    main()
