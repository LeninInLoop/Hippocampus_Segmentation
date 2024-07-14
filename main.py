import os.path
from src.train import Train, Validation
from src.data import *
from src.models import UNet3D, UNet3DVisualizer


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
    # Add more optimizer options here if needed
    raise ValueError(f"Unsupported optimizer: {Config.OPTIMIZER}")


def setup_data_loaders():
    """Set up and return data loaders."""
    hippocampus_strategy = VanderbiltHippocampusDatasetStrategy()
    hippocampus_factory = DefaultDataLoaderFactory(
        dataset_strategy=hippocampus_strategy,
        transform=train_transform,
        split_ratio=Config.TRAIN_SPLIT_RATIO,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS
    )
    hippocampus_loader = DataSetLoader(hippocampus_factory)

    return hippocampus_loader.get_train_loader(), hippocampus_loader.get_val_loader()


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
    # optimizer = get_optimizer(model)
    #
    # # Setup data loaders
    # train_loader, val_loader = setup_data_loaders()
    #
    # # Print batch info
    # print_batch_info(train_loader, "Train")
    # print_batch_info(val_loader, "Val")
    #
    # # Initialize and start training
    # trainer = Train(model, device, train_loader, val_loader, optimizer)
    # trainer.start_training()
    #
    # Validation.load_and_validate(model, Config.BEST_MODEL_SAVE_PATH, val_loader, device)
    visualizer = UNet3DVisualizer()
    visualizer.generate_diagram((16, 1, 48, 64, 48), filename="ModelDiagram", format="png")


if __name__ == '__main__':
    main()
