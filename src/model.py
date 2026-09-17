import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Residual Block
# ============================================================

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=8,
            num_channels=out_channels,
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.skip = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):

        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + residual
        x = self.act(x)

        return x


# ============================================================
# Image Encoder
#
# Input:
#     (B, 3, 192, 192)
#
# Output:
#     (B, 128, 12, 12)
#
# ============================================================

class ImageEncoder(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()

        self.input_block = ResBlock(
            in_channels,
            32,
        )

        # 192 -> 96
        self.down1 = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            ResBlock(32, 32),
        )

        # 96 -> 48
        self.down2 = nn.Sequential(
            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            ResBlock(64, 64),
        )

        # 48 -> 24
        self.down3 = nn.Sequential(
            nn.Conv2d(
                64,
                96,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            ResBlock(96, 96),
        )

        # 24 -> 12
        self.down4 = nn.Sequential(
            nn.Conv2d(
                96,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.GELU(),
            ResBlock(128, 128),
        )

        self.latent_norm = nn.GroupNorm(
            num_groups=8,
            num_channels=128,
        )

    def forward(self, x):

        x = self.input_block(x)

        x = self.down1(x)   # 192 -> 96
        x = self.down2(x)   # 96 -> 48
        x = self.down3(x)   # 48 -> 24
        x = self.down4(x)   # 24 -> 12

        z = self.latent_norm(x)

        return z

# ============================================================
# Parameter -> Latent Predictor
#
# Input:
#     (B, 2)
#
# Output:
#     (B, 128, 12, 12)
#
# ============================================================

class ParameterEncoder(nn.Module):

    def __init__(
        self,
        latent_channels=128,
        base_h=12,
        base_w=12,
    ):

        super().__init__()

        self.latent_channels = latent_channels
        self.base_h = base_h
        self.base_w = base_w

        latent_size = (
            latent_channels
            * base_h
            * base_w
        )

        self.mlp = nn.Sequential(

            nn.Linear(2, 128),
            nn.GELU(),

            nn.Linear(128, 256),
            nn.GELU(),

            nn.Linear(256, 512),
            nn.GELU(),

            nn.Linear(512, latent_size),
        )

    def forward(self, x):

        z = self.mlp(x)

        z = z.view(
            x.shape[0],
            self.latent_channels,
            self.base_h,
            self.base_w,
        )

        return z

# ============================================================
# Decoder
#
# Input:
#     (B, 128, 12, 12)
#
# Output:
#     (B, 3, 192, 192)
#
# ============================================================

class ResNetDecoder(nn.Module):

    def __init__(
        self,
        out_channels=3
    ):
        super().__init__()

        # ----------------------------------------------------
        # 12 x 12
        # ----------------------------------------------------

        self.bottleneck = nn.Sequential(
            ResBlock(128, 128),
            ResBlock(128, 128),
        )

        # ----------------------------------------------------
        # 12 x 12 -> 24 x 24
        # ----------------------------------------------------

        self.up1 = nn.ConvTranspose2d(
            128,
            96,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.res1 = nn.Sequential(
            ResBlock(96, 96),
            ResBlock(96, 96),
        )

        # ----------------------------------------------------
        # 24 x 24 -> 48 x 48
        # ----------------------------------------------------

        self.up2 = nn.ConvTranspose2d(
            96,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.res2 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )

        # ----------------------------------------------------
        # 48 x 48 -> 96 x 96
        # ----------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.res3 = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 32),
        )

        # ----------------------------------------------------
        # 96 x 96 -> 192 x 192
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.res4 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 16),
        )

        # ----------------------------------------------------
        # Final image: 192 x 192
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            16,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, z):

        # 12 -> 24
        z = self.bottleneck(z)

        z = self.up1(z)
        z = F.gelu(z)
        z = self.res1(z)

        # 24 -> 48
        z = self.up2(z)
        z = F.gelu(z)
        z = self.res2(z)

        # 48 -> 96
        z = self.up3(z)
        z = F.gelu(z)
        z = self.res3(z)

        # 96 -> 192
        z = self.up4(z)
        z = F.gelu(z)
        z = self.res4(z)

        image = self.output(z)

        return image


    
# ============================================================
# Complete Conditional Surrogate
# ============================================================

class ConditionalUResNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3
    ):

        super().__init__()

        # ----------------------------------------------------
        # Image -> latent
        #
        # (B, 3, 192, 192)
        # ->
        # (B, 128, 12, 12)
        #
        # Used only during training
        # ----------------------------------------------------

        self.image_encoder = ImageEncoder(
            in_channels=in_channels,
        )

        # ----------------------------------------------------
        # Parameters -> latent
        #
        # (B, 2)
        # ->
        # (B, 128, 12, 12)
        #
        # Used during training and inference
        # ----------------------------------------------------

        self.parameter_encoder = ParameterEncoder(
            latent_channels=128,
            base_h=12,
            base_w=12,
        )

        # ----------------------------------------------------
        # Latent -> image
        #
        # (B, 128, 12, 12)
        # ->
        # (B, 3, 192, 192)
        # ----------------------------------------------------

        self.decoder = ResNetDecoder(
            out_channels=out_channels
        )

    def encode_image(self, image):

        return self.image_encoder(image)

    def encode_parameters(self, x):

        return self.parameter_encoder(x)

    def decode(self, z):

        return self.decoder(z)

    def predict(self, x):

        z = self.encode_parameters(x)

        image = self.decode(z)

        return image

    def forward(self, x, image=None):

        z_pred = self.encode_parameters(x)

        image_pred = self.decode(z_pred)

        if image is None:
            return image_pred

        z_target = self.encode_image(image)

        return image_pred, z_pred, z_target


