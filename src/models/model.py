from src.models import Module, DownBlock, UpBlock, Conv2d


class UNet(Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()

        # Encoder
        self.e11 = DownBlock(3, 64)
        self.e21 = DownBlock(64, 128)
        self.e31 = DownBlock(128, 256)
        self.e41 = DownBlock(256, 512)
        self.e51 = DownBlock(512, 1024)

        # Decoder
        self.upconv1 = UpBlock(1024, 512)
        self.upconv2 = UpBlock(512, 256)
        self.upconv3 = UpBlock(256, 128)
        self.upconv4 = UpBlock(128, 64)

        # Output layer
        self.outconv = Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        x_e11, x_p1 = self.e11(x)
        x_e21, x_p2 = self.e21(x_e11)
        x_e31, x_p3 = self.e31(x_e21)
        x_e41, x_p4 = self.e41(x_e31)
        x_e51, x_p5 = self.e51(x_e41)

        # Decoder
        x_u1 = self.upconv1(x_e51, x_p4)
        x_d11 = self.upconv2(x_u1, x_p3)
        x_d21 = self.upconv3(x_d11, x_p2)
        x_d31 = self.upconv4(x_d21, x_p1)

        # Output layer
        out = self.outconv(x_d31)

        return out
