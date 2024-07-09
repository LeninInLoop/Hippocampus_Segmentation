from src.Config import Config
from src.data import *


def pad_3d_image(image, zero_pad=False, pad_ref=Config.PADDING_TARGET_SHAPE):
    """
    Pads a 3D image tensor to a specified reference size, choosing between padding with zeros or the minimum value of the image.

    Parameters:
    - image: A 4D PyTorch tensor representing the 3D image to be padded.
    - zero_pad: Boolean indicating whether to pad with zeros or the minimum value of the image.
    - pad_ref: Tuple specifying the reference size for padding. Default is (0, 0, 0), meaning no padding.

    Returns:
    - A 4D PyTorch tensor representing the padded image.
    """
    if zero_pad:
        value_to_pad = 0
    else:
        value_to_pad = image.min()

    target_shape = tuple(max(dim, ref) for dim, ref in zip(image.shape[1:], pad_ref))

    if value_to_pad == 0:
        image_padded = torch.zeros((*image.shape[:1], *target_shape))
    else:
        image_padded = value_to_pad * torch.ones((*image.shape[:1], *target_shape))

    image_padded[:, :image.shape[1], :image.shape[2], :image.shape[3]] = image

    return image_padded
