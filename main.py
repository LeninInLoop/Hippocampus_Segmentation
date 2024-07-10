from src.utils import *
from src.train import Train
from src.Config import Config
from src.data import dataset
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
    # Add more optimizer options here if needed
    raise ValueError(f"Unsupported optimizer: {Config.OPTIMIZER}")


def main():
    if not Config.USE_GPU:
        print("GPU usage is disabled in config. Using CPU instead.")
        device = torch.device("cpu")
    else:
        device = setup_gpu()
        if device is None:
            return

    print_config()
    print_dataset_info()

    model = initialize_model(device)
    optimizer = get_optimizer(model)

    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()

    print_batch_info(train_loader, "Train")
    print_batch_info(val_loader, "Val")

    trainer = Train(model, device, train_loader, val_loader, optimizer)
    trainer.start_training()


if __name__ == '__main__':
    main()

