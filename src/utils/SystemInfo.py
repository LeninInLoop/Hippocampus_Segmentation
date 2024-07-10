import torch
from typing import Optional, List, Dict

class SystemInfo:
    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()

    @staticmethod
    def get_cuda_device_count() -> int:
        """Get the number of available CUDA devices."""
        return torch.cuda.device_count() if SystemInfo.is_cuda_available() else 0

    @staticmethod
    def get_current_cuda_device() -> Optional[int]:
        """Get the current CUDA device index."""
        return torch.cuda.current_device() if SystemInfo.is_cuda_available() else None

    @staticmethod
    def set_cuda_device(device_idx: int) -> bool:
        """
        Set the current CUDA device.

        Args:
            device_idx (int): The index of the device to set.

        Returns:
            bool: True if the device was set successfully, False otherwise.
        """
        if not SystemInfo.is_cuda_available():
            print(f"CUDA is not available for device {device_idx}")
            return False

        try:
            torch.cuda.set_device(device_idx)
            return True
        except RuntimeError:
            print(f"Failed to set CUDA device {device_idx}")
            return False

    @staticmethod
    def get_cuda_devices() -> List[Dict]:
        """Get information about all available CUDA devices."""
        return [SystemInfo._get_cuda_device_info(i) for i in range(SystemInfo.get_cuda_device_count())]

    @staticmethod
    def _get_cuda_device_info(device_idx: int) -> Dict:
        """
        Get information about a specific CUDA device.

        Args:
            device_idx (int): The index of the device.

        Returns:
            dict: A dictionary containing device information.
        """
        properties = torch.cuda.get_device_properties(device_idx)
        return {
            'device_id': device_idx,
            'name': properties.name,
            'memory': properties.total_memory,
            'compute_capability': properties.major * 10 + properties.minor,
        }

    @staticmethod
    def get_device_with_highest_compute_capability() -> int:
        """Get the index of the device with the highest compute capability."""
        devices = SystemInfo.get_cuda_devices()
        return max(devices, key=lambda d: d['compute_capability'])['device_id'] if devices else -1

    @staticmethod
    def get_device_with_highest_memory() -> int:
        """Get the index of the device with the highest memory."""
        devices = SystemInfo.get_cuda_devices()
        return max(devices, key=lambda d: d['memory'])['device_id'] if devices else -1
