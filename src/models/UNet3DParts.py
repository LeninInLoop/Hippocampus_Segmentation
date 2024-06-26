import torch
import torch.nn as nn


def InitialConvolutionalLayer(in_channels, middle_channel, out_channels):
    """
    Creates an initial convolutional layer with two 3D convolutions.

    This layer follows the strategy:
    (in_channels -> middle_channel -> out_channels)

    Benefits:
    - Allows for a gradual increase in the number of channels
    - Provides an initial feature extraction stage
    - Can help in learning low-level features before more complex transformations

    Args:
        in_channels (int): Number of input channels (typically the number of image channels)
        middle_channel (int): Number of channels after the first convolution
        out_channels (int): Number of output channels

    Returns:
        nn.Sequential: A PyTorch Sequential module representing the layer
    """
    conv = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=middle_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels=middle_channel, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def DownSample():
    """
    Creates a down-sampling layer using max pooling.

    This layer reduces the spatial dimensions of the input by half in each dimension.
    It's typically used in the contracting path of a U-Net-like architecture.

    Returns:
        nn.MaxPool3d: A PyTorch 3D Max Pooling layer
    """
    return nn.MaxPool3d(kernel_size=2, stride=2)


def UpSample(in_channels, out_channels):
    """
    Creates an up-sampling layer using transposed convolution.

    This layer increases the spatial dimensions of the input by a factor of 2 in each dimension.
    It's typically used in the expanding path of a U-Net-like architecture.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Returns:
        nn.ConvTranspose3d: A PyTorch 3D Transposed Convolutional layer
    """
    return nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)


def DownConvolutionalLayer(in_channels, out_channels):
    """
    Creates a down-convolutional layer with two 3D convolutions.

    Two implementation strategies are possible:

    1. First Implementation (in_channels -> in_channels -> out_channels):
       - Preserves more input information in the intermediate layer
       - Beneficial for processing input before reducing dimensionality
       - Potentially learns more complex features before channel reduction

    2. Second Implementation (in_channels -> out_channels -> out_channels):
       - Reduces channels earlier, improving computational efficiency
       - May help reduce overfitting by limiting model capacity earlier
       - Beneficial for quick transformation to a different feature space

    This function uses the Second Implementation.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Returns:
        nn.Sequential: A PyTorch Sequential module representing the layer
    """
    conv = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv

# Example of the First Implementation (commented out):
#
# def DownConvolutionalLayer(in_channels, out_channels):
#     conv = nn.Sequential(
#         nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
#         nn.ReLU(inplace=False),
#         nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
#         nn.ReLU(inplace=False)
#     )
#     return conv


def UpConvolutionalLayer(in_channels, out_channels):
    """
    Creates an up-convolutional layer with two 3D convolutions.

    This implementation follows the strategy:
    (in_channels -> out_channels -> out_channels)

    Benefits:
    - Reduces channels early, potentially improving computational efficiency
    - May help in reducing overfitting by limiting the model's capacity earlier
    - Useful for quickly transforming the input to the desired feature space

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels

    Returns:
        nn.Sequential: A PyTorch Sequential module representing the layer
    """
    conv = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def FinalConvolutionalLayer(in_channels, out_channels):
    """
    Creates a final convolutional layer with a 1x1x1 kernel.

    This layer is typically used to map the feature channels to the desired output channels,
    often corresponding to the number of classes in a segmentation task.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (often the number of classes)

    Returns:
        nn.Conv3d: A PyTorch 3D Convolutional layer
    """
    return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


def ConcatBlock(x1, x2):
    """
    Concatenates two tensors along the channel dimension.

    This function is typically used in skip connections, where features from an earlier layer
    are combined with upsampled features from a later layer.

    Args:
        x1 (torch.Tensor): First tensor to concatenate
        x2 (torch.Tensor): Second tensor to concatenate

    Returns:
        torch.Tensor: Concatenated tensor
    """
    return torch.cat((x1, x2), 1)
