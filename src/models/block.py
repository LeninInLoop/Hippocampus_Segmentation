from src.models import Module, Conv2d, ReLU


class Block(Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)))
