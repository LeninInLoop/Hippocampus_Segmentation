from src.utils import SystemInfo, print_dataset_info, print_cuda_device_info, System
from src.Config import Config
from src.data import DataSetLoader

import torch


def main():
    if Config.USE_GPU:
        if not SystemInfo.is_cuda_available():
            return print("No GPU is Available")
        print("CUDA is available.")

        cuda_devices = SystemInfo.get_cuda_devices()
        print_cuda_device_info(cuda_devices)

        if Config.USE_GPU_WITH_MORE_MEMORY:
            System.set_cuda_device_with_highest_mem(cuda_devices)

        if Config.USE_GPU_WITH_MORE_COMPUTE_CAPABILITY:
            System.set_cuda_device_with_highest_compute(cuda_devices)

        # train_data, val_data, test_data = HippocampusDataset.get_data_loaders()
        # print(train_data)
        # # Iterate over the train loader
        # for batch_images, batch_labels in train_data:
        #     print(f"Batch Images Shape: {batch_images.shape}")
        #     print(f"Batch Labels Shape: {batch_labels.shape}")


if __name__ == '__main__':
    main()

    dataset = DataSetLoader()
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
    # with open(Config.DATA_JSON, 'r') as f:
    #     data_json = json.load(f)
    # data_list = data_json['training']
    # for idx in range(len(data_list)):
    #     img_path = Config.DATA_DIR + data_list[idx]['image'][1:]
    #     print(img_path)
    #     image = nib.load(img_path)
    #     print(image.shape)
