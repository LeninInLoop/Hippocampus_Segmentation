from src.utils import SystemInfo
from src.config import Config
from src.models.model import UNet3D
import torch


def main():
    if Config.USE_GPU:
        if not SystemInfo.is_cuda_available():
            return print("No GPU is Available")
        print("CUDA is available.")

        cuda_devices = SystemInfo.get_cuda_devices()
        print("CUDA devices are shown below:")
        print("----------------------------------------------------------")
        for index, device in enumerate(cuda_devices):
            print(f"Device {index + 1}/{len(cuda_devices)}")
            print(f"Device ID: {device['device_id']}")
            print(f"Device Name: {device['name']}")
            print(f"Device Memory: {device['memory']}")
            print(f"Device Compute Capability: {device['compute_capability']}")
        print("----------------------------------------------------------\n")

        if Config.USE_GPU_WITH_MORE_MEMORY:
            if len(cuda_devices) > 1:
                device_with_more_memory = SystemInfo.get_device_with_highest_memory()
                print(f"Device with highest memory: {device_with_more_memory}")

                print(f"Setting current CUDA device to: {device_with_more_memory}")
                SystemInfo.set_cuda_device(device_with_more_memory)

        if Config.USE_GPU_WITH_MORE_COMPUTE_CAPABILITY:
            if len(cuda_devices) > 1:
                device_with_highest_compute = SystemInfo.get_device_with_highest_compute_capability()
                print(f"Device with highest compute capability: {device_with_highest_compute}")

                print(f"Setting current CUDA device to: {device_with_highest_compute}")
                SystemInfo.set_cuda_device(device_with_highest_compute)

        input_example = torch.rand((4, 1, 64, 64, 64))
        Unet3D = UNet3D(in_channels=1, out_channels=2, feat_channels=32)
        output = Unet3D(input_example)
        print(output.shape)


if __name__ == '__main__':
    main()
