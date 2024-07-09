from src.train import *
from src.utils import *
from src.Config import Config
from src.data import dataset
from src.models import UNet3D


def main():
    global optimizer
    if Config.USE_GPU:
        if not SystemInfo.is_cuda_available():
            return print("No GPU is Available")
        print("CUDA is available.")

        cuda_devices = SystemInfo.get_cuda_devices()
        print_cuda_device_info(cuda_devices)

        device = torch.device("cuda")

        if Config.USE_GPU_WITH_MORE_MEMORY:
            System.set_cuda_device_with_highest_mem(cuda_devices)

        if Config.USE_GPU_WITH_MORE_COMPUTE_CAPABILITY:
            System.set_cuda_device_with_highest_compute(cuda_devices)

        device = torch.device("cuda")

        # Initialize the model
        model = UNet3D(in_channels=1, out_channels=3, feat_channels=32).to(device)

        # Define loss function
        criterion = get_multi_dice_loss()

        # Define optimizer
        if Config.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        elif Config.OPTIMIZER == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM,
                                  weight_decay=Config.WEIGHT_DECAY)
        else:
            raise ValueError(f"Unsupported optimizer: {Config.OPTIMIZER}")

        # Define learning rate scheduler
        if Config.LR_SCHEDULER == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
        elif Config.LR_SCHEDULER == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=Config.LR_PATIENCE,
                                                             factor=Config.LR_GAMMA)
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {Config.LR_SCHEDULER}")

        # Get data loaders
        train_loader = dataset.get_train_loader()
        val_loader = dataset.get_val_loader()

        # Print sample batch shapes
        for batch_images, batch_labels in train_loader:
            print(f"Train batch image shape: {batch_images.shape}")
            print(f"Train batch label shape: {batch_labels.shape}")
            break

        for batch_images, batch_labels in val_loader:
            print(f"Val batch image shape: {batch_images.shape}")
            print(f"Val batch label shape: {batch_labels.shape}")
            break

        print(f"Number of training samples: {len(train_loader.dataset)}")
        print(f"Number of validation samples: {len(val_loader.dataset)}")

        # Create Train instance and start training
        trainer = Train(model, device, train_loader, val_loader, optimizer, criterion, scheduler)
        trainer.start_training()


if __name__ == '__main__':
    main()

    # with open(Config.DATA_JSON, 'r') as f:
    #     data_json = json.load(f)
    # data_list = data_json['training']
    # for idx in range(len(data_list)):
    #     img_path = Config.DATA_DIR + data_list[idx]['image'][1:]
    #     print(img_path)
    #     image = nib.load(img_path)
    #     print(image.shape)
