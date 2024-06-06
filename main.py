from src.models import UNet
from src.utils import SystemInfo
import test


def main():
    device_info = SystemInfo.get_cuda_device_info()
    print("CUDA is available. Number of CUDA devices:", SystemInfo.get_cuda_device_count())
    print(device_info)
    print(SystemInfo.get_current_cuda_device())
    print(SystemInfo.set_cuda_device(0))
    print(SystemInfo.set_cuda_device(1))
    print(SystemInfo.get_device_with_highest_compute_capability())
    test.run(UNet)


if __name__ == '__main__':
    main()
