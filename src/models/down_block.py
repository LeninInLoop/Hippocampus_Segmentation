from src.models import Module, Block, MaxPool2d


class DownBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.block = Block(in_channels, out_channels)
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_block, x_pooled = self.block(x), self.pool(x)
        return x_block, x_pooled
