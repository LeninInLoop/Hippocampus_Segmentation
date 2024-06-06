from src.models import torch


class SystemInfo:
    @staticmethod
    def is_cuda_available() -> bool:
        return bool(torch.cuda.is_available())

    @staticmethod
    def get_cuda_device_count() -> int:
        if SystemInfo.is_cuda_available():
            return torch.cuda.device_count()
        else:
            return 0

    @staticmethod
    def get_current_cuda_device() -> int | None:
        if SystemInfo.is_cuda_available():
            return torch.cuda.current_device()
        else:
            return None

    @staticmethod
    def set_cuda_device(device_idx) -> bool:
        if SystemInfo.is_cuda_available():
            try:
                torch.cuda.set_device(device_idx)
                return True
            except RuntimeError:
                print(f"CUDA is not available for device {device_idx}")
                return False
        else:
            print(f"CUDA is not available for device {device_idx}")
            return False

    @staticmethod
    def get_cuda_device_info() -> list:
        device_info = []
        if SystemInfo.is_cuda_available():
            for i in range(SystemInfo.get_cuda_device_count()):
                device_info.append(SystemInfo._get_cuda_device_info(i))
        return device_info

    @staticmethod
    def _get_cuda_device_info(device_idx) -> dict:
        if SystemInfo.is_cuda_available():
            properties = torch.cuda.get_device_properties(device_idx)
            return {
                'device_id': device_idx,
                'name': properties.name,
                'memory': properties.total_memory,
                'compute_capability': properties.major * 10 + properties.minor,
            }

    @staticmethod
    def get_device_with_highest_compute_capability() -> int:
        max_compute_capability = 0
        device_idx_with_max_compute_capability = -1

        device_info = SystemInfo.get_cuda_device_info()
        for device in device_info:
            compute_capability = device['compute_capability']
            if compute_capability > max_compute_capability:
                max_compute_capability = compute_capability
                device_idx_with_max_compute_capability = device['device_id']

        return device_idx_with_max_compute_capability
