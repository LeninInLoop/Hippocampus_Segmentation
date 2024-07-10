def print_dataset_info(data_json):
    """
    Print formatted information about the dataset.

    Args:
        data_json (dict): A dictionary containing dataset information.
    """
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
        print(f"{label}: {value}")
    print()


def print_cuda_device_info(cuda_devices):
    """
    Print formatted information about CUDA devices.

    Args:
        cuda_devices (list): A list of dictionaries containing CUDA device information.
    """
    print("CUDA devices are shown below:")
    print("-" * 58)

    for index, device in enumerate(cuda_devices, 1):
        print(f"Device {index}/{len(cuda_devices)}")
        device_info = [
            ("Device ID", device['device_id']),
            ("Device Name", device['name']),
            ("Device Memory", device['memory']),
            ("Device Compute Capability", device['compute_capability'])
        ]
        for label, value in device_info:
            print(f"  {label}: {value}")
        if index < len(cuda_devices):
            print()  # Add a blank line between devices, except for the last one

    print("-" * 58)
    print()
