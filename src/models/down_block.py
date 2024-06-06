from src.models import Module, Block, MaxPool2d


class DownBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = Block(in_channels, out_channels)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.block(x)
        p = self.pool(x)
        return x, p
