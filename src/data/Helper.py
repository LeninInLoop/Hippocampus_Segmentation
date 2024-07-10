from src.Config import Config
from src.data import *


def get_pad_3d_image(pad_ref: tuple = Config.PADDING_TARGET_SHAPE, zero_pad: bool = True):
    """
    Creates a function to pad 3D images to a specified size.

    Args:
        pad_ref (tuple): The target size for padding (depth, height, width). Default is (48, 64, 48).
        zero_pad (bool): If True, pad with zeros. If False, pad with the minimum value of the image. Default is True.

    Returns:
        function: A function that pads 3D images to the specified size.
    """

    def pad_3d_image(image: torch.Tensor) -> torch.Tensor:
        """
        Pads a 3D image tensor to the specified size.

        Args:
            image (torch.Tensor): The input 3D image tensor.

        Returns:
            torch.Tensor: The padded 3D image tensor.
        """
        # Determine the value to use for padding
        value_to_pad = 0 if zero_pad else image.min()

        # Create the target shape, including the channel dimension
        pad_ref_channels = (image.shape[0], *pad_ref)

        # Create the padded image tensor
        if value_to_pad == 0:
            image_padded = torch.zeros(pad_ref_channels)
        else:
            image_padded = torch.full(pad_ref_channels, value_to_pad)

        # Copy the original image into the padded tensor
        image_padded[:, :image.shape[1], :image.shape[2], :image.shape[3]] = image

        return image_padded

    return pad_3d_image

# Usage example:
# pad_func = get_pad_3d_image(pad_ref=(128, 128, 128), zero_pad=False)
# padded_image = pad_func(original_image)
