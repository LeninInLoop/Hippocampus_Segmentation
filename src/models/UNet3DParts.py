import torch
import torch.nn as nn


def InitialConvolutionalLayer(in_channels, middle_channel, out_channels):
    conv = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=middle_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels=middle_channel, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def DownSample():
    return nn.MaxPool3d(kernel_size=2, stride=2)


def UpSample(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)


def DownConvolutionalLayer(in_channels, out_channels):
    # First Implementation (in_channels -> in_channels -> out_channels):
    #     Preserves more information from the input in the intermediate layer.
    #     May be beneficial when you want to process the input more before reducing dimensionality.
    #     Could potentially learn more complex features before reducing the number of channels.
    #
    # conv = nn.Sequential(
    #     nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=False),
    #     nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
    #     nn.ReLU(inplace=False)
    # )
    # return conv

    # Second Implementation (in_channels -> out_channels -> out_channels):
    #     Reduces the number of channels earlier, which can be computationally more efficient.
    #     May help in reducing overfitting by limiting the model's capacity earlier in the layer.
    #     Could be beneficial when you want to quickly transform the input to a different feature space.
    conv = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def UpConvolutionalLayer(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=False)
    )
    return conv


def FinalConvolutionalLayer(in_channels, out_channels):
    return nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


def ConcatBlock(x1, x2):
    return torch.cat((x1, x2), 1)
