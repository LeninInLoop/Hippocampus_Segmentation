from src.models import Module, ReLU, Conv2d, Sequential, EncoderBlock, DecoderBlock


class UNet(Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.center = Sequential(
            Conv2d(512, 1024, kernel_size=3, padding=1),
            ReLU(inplace=True),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            ReLU(inplace=True),
        )

        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)

        self.outconv = Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1, p1 = self.enc1(x)
        e2, p2 = self.enc2(p1)
        e3, p3 = self.enc3(p2)
        e4, p4 = self.enc4(p3)

        # Center
        center = self.center(p4)

        # Decoder
        d1 = self.dec1(center, e4)
        d2 = self.dec2(d1, e3)
        d3 = self.dec3(d2, e2)
        d4 = self.dec4(d3, e1)

        # Output layer
        out = self.outconv(d4)

        return out
