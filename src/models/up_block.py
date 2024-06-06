from src.models import Module, Conv2d, MaxPool2d, functional


class EncoderBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        p = self.pool(x)
        return x, p
