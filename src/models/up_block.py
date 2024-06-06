from src.models import Module, ConvTranspose2d, Block, torch


class UpBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.block = Block(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x
