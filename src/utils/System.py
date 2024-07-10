from .SystemInfo import SystemInfo


class System:
    @staticmethod
    def set_cuda_device_with_highest_mem(cuda_devices):
        """
        Set the CUDA device with the highest memory as the current device.

        Args:
            cuda_devices (list): List of available CUDA devices.

        Returns:
            bool: True if a device was set, False otherwise.
        """
        return System._set_cuda_device_by_criteria(
            cuda_devices,
            SystemInfo.get_device_with_highest_memory,
            "highest memory"
        )

    @staticmethod
    def set_cuda_device_with_highest_compute(cuda_devices):
        """
        Set the CUDA device with the highest compute capability as the current device.

        Args:
            cuda_devices (list): List of available CUDA devices.

        Returns:
            bool: True if a device was set, False otherwise.
        """
        return System._set_cuda_device_by_criteria(
            cuda_devices,
            SystemInfo.get_device_with_highest_compute_capability,
            "highest compute capability"
        )

    @staticmethod
    def _set_cuda_device_by_criteria(cuda_devices, get_device_func, criteria_description):
        """
        Helper method to set CUDA device based on a given criteria.

        Args:
            cuda_devices (list): List of available CUDA devices.
            get_device_func (callable): Function to get the device based on criteria.
            criteria_description (str): Description of the criteria for logging.

        Returns:
            bool: True if a device was set, False otherwise.
        """
        if len(cuda_devices) <= 1:
            return False

        selected_device = get_device_func()
        print(f"Device with {criteria_description}: {selected_device}")
        print(f"Setting current CUDA device to: {selected_device}")

        return SystemInfo.set_cuda_device(selected_device)