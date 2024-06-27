
def print_dataset_info(data_json):
    print(f"Dataset Name: {data_json['name']}")
    print(f"Description: {data_json['description']}")
    print(f"Reference: {data_json['reference']}")
    print(f"Licence: {data_json['licence']}")
    print(f"Release Version: {data_json['relase']}")
    print(f"Tensor Image Size: {data_json['tensorImageSize']}")
    print(f"Modality: {data_json['modality']['0']}")
    print(f"Number of Training Images: {data_json['numTraining']}")
    print(f"Number of Test Images: {data_json['numTest']}\n")


def print_cuda_device_info(cuda_devices):
    print("CUDA devices are shown below:")
    print("----------------------------------------------------------")
    for index, device in enumerate(cuda_devices):
        print(f"Device {index + 1}/{len(cuda_devices)}")
        print(f"Device ID: {device['device_id']}")
        print(f"Device Name: {device['name']}")
        print(f"Device Memory: {device['memory']}")
        print(f"Device Compute Capability: {device['compute_capability']}")
    print("----------------------------------------------------------\n")
