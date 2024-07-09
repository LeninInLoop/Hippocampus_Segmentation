from src.models.UNet3DParts import *


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, feat_channels=32):
        super().__init__()

        # Encoder Block
        self.down_sample = DownSample()

        self.down_conv1 = InitialConvolutionalLayer(in_channels, feat_channels, feat_channels * 2)
        self.down_conv2 = DownConvolutionalLayer(feat_channels * 2, feat_channels * 4)
        self.down_conv3 = DownConvolutionalLayer(feat_channels * 4, feat_channels * 8)
        self.down_conv4 = DownConvolutionalLayer(feat_channels * 8, feat_channels * 16)

        # Decoder Block
        self.up_sample1 = UpSample(feat_channels * 16, feat_channels * 16)
        self.up_conv1 = UpConvolutionalLayer(feat_channels * (16 + 8), feat_channels * 8)

        self.up_sample2 = UpSample(feat_channels * 8, feat_channels * 8)
        self.up_conv2 = UpConvolutionalLayer(feat_channels * (8 + 4), feat_channels * 4)

        self.up_sample3 = UpSample(feat_channels * 4, feat_channels * 4)
        self.up_conv3 = UpConvolutionalLayer(feat_channels * (4 + 2), feat_channels * 2)

        # Output layer
        self.final_conv = FinalConvolutionalLayer(feat_channels * 2, out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        init_layer = self.down_conv1(image)
        down_sample1 = self.down_sample(init_layer)

        down_conv2 = self.down_conv2(down_sample1)
        down_sample2 = self.down_sample(down_conv2)

        down_conv3 = self.down_conv3(down_sample2)
        down_sample3 = self.down_sample(down_conv3)

        down_conv4 = self.down_conv4(down_sample3)
        up_sample1 = self.up_sample1(down_conv4)

        concat1 = ConcatBlock(up_sample1, down_conv3)
        up_conv1 = self.up_conv1(concat1)
        up_sample2 = self.up_sample2(up_conv1)

        concat2 = ConcatBlock(up_sample2, down_conv2)
        up_conv2 = self.up_conv2(concat2)
        up_sample3 = self.up_sample3(up_conv2)

        concat3 = ConcatBlock(up_sample3, init_layer)
        up_conv3 = self.up_conv3(concat3)

        final_conv = self.final_conv(up_conv3)
        return self.softmax(final_conv)


# Test

