import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Fourier / Positional Features
# ============================================================

def make_fourier_grid(h, w, num_freqs, device):

    ys = torch.linspace(-1, 1, h, device=device)
    xs = torch.linspace(-1, 1, w, device=device)

    grid_y, grid_x = torch.meshgrid(
        ys,
        xs,
        indexing="ij",
    )

    feats = []

    for i in range(num_freqs):

        freq = 2.0 ** i

        feats += [
            torch.sin(freq * np.pi * grid_x),
            torch.cos(freq * np.pi * grid_x),
            torch.sin(freq * np.pi * grid_y),
            torch.cos(freq * np.pi * grid_y),
        ]

    return torch.stack(feats, dim=0)


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
#     (B, 3, 288, 288)
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

        # 288 -> 144
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

        # 144 -> 72
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

        # 72 -> 36
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

        # 36 -> 12
        #
        # Instead of another factor-of-2 downsampling,
        # directly map 36 x 36 -> 12 x 12.
        self.down4 = nn.Sequential(

            nn.Conv2d(
                96,
                128,
                kernel_size=4,
                stride=3,
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

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

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
# Fourier Positional Injection
# ============================================================

class FourierInjection(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_freqs=4,
    ):

        super().__init__()

        self.num_freqs = num_freqs

        # Each frequency contributes:
        #
        # sin(x), cos(x), sin(y), cos(y)
        #
        # => 4 channels per frequency
        fourier_channels = 4 * num_freqs

        self.proj = nn.Conv2d(
            in_channels + fourier_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        B, C, H, W = x.shape

        fourier = make_fourier_grid(
            H,
            W,
            self.num_freqs,
            x.device,
        )

        # (4*num_freqs, H, W)
        fourier = fourier.unsqueeze(0)

        # (B, 4*num_freqs, H, W)
        fourier = fourier.expand(
            B,
            -1,
            -1,
            -1,
        )

        x = torch.cat(
            [x, fourier],
            dim=1,
        )

        x = self.proj(x)

        return x


# ============================================================
# Decoder
#
# Input:
#     (B, 128, 12, 12)
#
# Output:
#     (B, 3, 288, 288)
#
# Fourier features are injected at every spatial resolution.
# ============================================================

class ResNetDecoder(nn.Module):

    def __init__(
        self,
        out_channels=3,
        num_freqs=4,
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
        # 12 x 12 -> 36 x 36
        # ----------------------------------------------------

        self.up1 = nn.ConvTranspose2d(
            128,
            96,
            kernel_size=3,
            stride=3,
            padding=0,
        )

        self.fourier1 = FourierInjection(
            96,
            96,
            num_freqs=num_freqs,
        )

        self.res1 = nn.Sequential(
            ResBlock(96, 96),
            ResBlock(96, 96),
        )

        # ----------------------------------------------------
        # 36 x 36 -> 72 x 72
        # ----------------------------------------------------

        self.up2 = nn.ConvTranspose2d(
            96,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.fourier2 = FourierInjection(
            64,
            64,
            num_freqs=num_freqs,
        )

        self.res2 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
        )

        # ----------------------------------------------------
        # 72 x 72 -> 144 x 144
        # ----------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.fourier3 = FourierInjection(
            32,
            32,
            num_freqs=num_freqs,
        )

        self.res3 = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 32),
        )

        # ----------------------------------------------------
        # 144 x 144 -> 288 x 288
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            32,
            16,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.fourier4 = FourierInjection(
            16,
            16,
            num_freqs=num_freqs,
        )

        self.res4 = nn.Sequential(
            ResBlock(16, 16),
            ResBlock(16, 16),
        )

        # ----------------------------------------------------
        # Final image
        # ----------------------------------------------------

        self.output = nn.Conv2d(
            16,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, z):

        # 12 x 12
        z = self.bottleneck(z)

        # ----------------------------------------------------
        # 12 -> 36
        # ----------------------------------------------------

        z = self.up1(z)
        z = F.gelu(z)

        z = self.fourier1(z)
        z = self.res1(z)

        # ----------------------------------------------------
        # 36 -> 72
        # ----------------------------------------------------

        z = self.up2(z)
        z = F.gelu(z)

        z = self.fourier2(z)
        z = self.res2(z)

        # ----------------------------------------------------
        # 72 -> 144
        # ----------------------------------------------------

        z = self.up3(z)
        z = F.gelu(z)

        z = self.fourier3(z)
        z = self.res3(z)

        # ----------------------------------------------------
        # 144 -> 288
        # ----------------------------------------------------

        z = self.up4(z)
        z = F.gelu(z)

        z = self.fourier4(z)
        z = self.res4(z)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        image = self.output(z)

        return image


# ============================================================
# Complete Conditional Surrogate
# ============================================================

class ConditionalUResNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_freqs=4,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Image -> latent
        #
        # (B, 3, 288, 288)
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
        # (B, 3, 288, 288)
        # ----------------------------------------------------

        self.decoder = ResNetDecoder(
            out_channels=out_channels,
            num_freqs=num_freqs,
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



    
# # ============================================================
# # Test
# # ============================================================

# if __name__ == "__main__":

#     model = ConditionalUResNet(
#         in_channels=1,
#         out_channels=1,
#     )

#     B = 4

#     x = torch.randn(B, 2)
#     image = torch.randn(B, 1, 96, 384)

#     # Training forward
#     image_pred, z_pred, z_target = model(
#         x,
#         image,
#     )

#     print("Input parameters :", x.shape)
#     print("Input image      :", image.shape)
#     print("Predicted latent :", z_pred.shape)
#     print("Target latent    :", z_target.shape)
#     print("Predicted image  :", image_pred.shape)

#     # Inference
#     output = model.predict(x)

#     print("Inference output :", output.shape)

#     n_params = sum(
#         p.numel()
#         for p in model.parameters()
#     )

#     print(f"Parameters       : {n_params:,}")