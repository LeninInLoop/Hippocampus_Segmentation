from src.models import Module, Conv2d, torch, ConvTranspose2d, functional


class DecoderBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        return x
