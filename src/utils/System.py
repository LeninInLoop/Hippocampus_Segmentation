from .SystemInfo import SystemInfo


class System:
    @staticmethod
    def set_cuda_device_with_highest_mem(cuda_devices) -> bool:
        if len(cuda_devices) > 1:
            device_with_more_memory = SystemInfo.get_device_with_highest_memory()
            print(f"Device with highest memory: {device_with_more_memory}")

            print(f"Setting current CUDA device to: {device_with_more_memory}")
            return SystemInfo.set_cuda_device(device_with_more_memory)
        return False

    @staticmethod
    def set_cuda_device_with_highest_compute(cuda_devices):
        if len(cuda_devices) > 1:
            device_with_highest_compute = SystemInfo.get_device_with_highest_compute_capability()
            print(f"Device with highest compute capability: {device_with_highest_compute}")

            print(f"Setting current CUDA device to: {device_with_highest_compute}")
            return SystemInfo.set_cuda_device(device_with_highest_compute)
        return False
