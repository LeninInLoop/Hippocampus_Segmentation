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

        # Initialize the model
        model = UNet3D().to(device)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        if Config.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        elif Config.OPTIMIZER == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM,
                                  weight_decay=Config.WEIGHT_DECAY)

        # Define learning rate scheduler
        if Config.LR_SCHEDULER == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
        elif Config.LR_SCHEDULER == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=Config.LR_PATIENCE,
                                                             factor=Config.LR_GAMMA)
        else:
            raise ValueError(f"Unsupported learning rate scheduler: {Config.LR_SCHEDULER}")

        train_loader = dataset.get_train_loader()

        for batch_images, batch_labels in train_loader:
            print(f"Batch image shape: {batch_images.shape}")
            print(f"Batch label shape: {batch_labels.shape}")
            break  # Just print the first batch and exit the loop

        print(len(train_loader.dataset))

        val_loader = dataset.get_val_loader()
        for batch_images, batch_labels in val_loader:
            print(f"Batch image shape: {batch_images.shape}")
            print(f"Batch label shape: {batch_labels.shape}")
            break  # Just print the first batch and exit the loop

        print(len(val_loader.dataset))

        unet3_d = UNet3D(in_channels=1, out_channels=2, feat_channels=32).cuda()

        # Iterate over the train_loader to get batches
        for batch_images, batch_labels in train_loader:
            # Move batch to GPU if using CUDA
            batch_images = batch_images.cuda()
            batch_labels = batch_labels.cuda()

            print("Unique values in batch_labels:", torch.unique(batch_labels))
            print("Minimum value in batch_labels:", batch_labels.min().item())
            print("Maximum value in batch_labels:", batch_labels.max().item())

            # Forward pass
            outputs = unet3_d(batch_images)

            print("Output shape =", outputs.shape)
            print("Batch labels shape =", batch_labels.shape)

            # Calculating loss using the custom dice loss
            loss = get_multi_dice_loss(outputs, batch_labels, device=batch_images.device)
            print("Dice Loss =", loss.item())

            # Back propagation
            loss.backward()

            # Here you would typically update the model parameters using an optimizer
            # optimizer.step()
            # optimizer.zero_grad()

            break  # Remove this if you want to process all batches

        # Create Train instance and start training
        # trainer = Train(model, device, train_loader, val_loader, optimizer, criterion, scheduler)
        # trainer.start_training()

        # train_data, val_data, test_data = HippocampusDataset.get_data_loaders()
        # print(train_data)
        # # Iterate over the train loader
        # for batch_images, batch_labels in train_data:
        #     print(f"Batch Images Shape: {batch_images.shape}")
        #     print(f"Batch Labels Shape: {batch_labels.shape}")


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
