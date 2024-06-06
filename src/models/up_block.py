from src.models import Module, ConvTranspose2d, Block, torch


class UpBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upconv = ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x_down, x_up):
        x_up = self.upconv(x_down)
        x_cat = torch.cat([x_up, x_down], dim=1)
        return x_cat