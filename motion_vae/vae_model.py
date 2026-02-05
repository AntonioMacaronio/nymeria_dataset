"""
Motion VAE model with 1D causal convolutions.

Based on EgoTwin paper (arXiv:2508.13013) architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import MotionVAEConfig
from .motion_preprocessing import NymeriaMotionPreprocessor, HeadCentricMotion


class CausalConv1d(nn.Module):
    """1D causal convolution with left padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        # Causal padding: (kernel_size - 1) * dilation on the left only
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # We handle padding manually
            dilation=dilation,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T)
        Returns:
            Output tensor of shape (B, C_out, T) for stride=1
        """
        # Left pad for causality
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResNetBlock1D(nn.Module):
    """1D ResNet block with causal convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size)
        self.conv2 = CausalConv1d(channels, channels, kernel_size)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T)
        Returns:
            Output tensor of shape (B, C, T)
        """
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + residual


class DownsampleBlock(nn.Module):
    """Downsampling block with stride-2 convolution for 2x temporal downsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Use kernel_size=4, stride=2 for smooth downsampling
        # Causal padding: (4-1)*1 = 3 on left
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=2, padding=0)
        self.padding = 3  # Causal padding

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T)
        Returns:
            Output tensor of shape (B, C_out, T//2)
        """
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Upsampling block with transposed convolution for 2x temporal upsampling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Transposed conv for upsampling
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, T)
        Returns:
            Output tensor of shape (B, C_out, T*2)
        """
        return self.conv(x)


class MotionEncoder(nn.Module):
    """Encoder for motion VAE with 4x temporal downsampling."""

    def __init__(self, config: MotionVAEConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.input_proj = CausalConv1d(config.input_dim, config.hidden_dim, kernel_size=3)

        # Stage 1: ResNet blocks + downsample
        self.stage1_blocks = nn.ModuleList([
            ResNetBlock1D(config.hidden_dim) for _ in range(config.num_resnet_blocks)
        ])
        self.downsample1 = DownsampleBlock(config.hidden_dim, config.hidden_dim)

        # Stage 2: ResNet blocks + downsample
        self.stage2_blocks = nn.ModuleList([
            ResNetBlock1D(config.hidden_dim) for _ in range(config.num_resnet_blocks)
        ])
        self.downsample2 = DownsampleBlock(config.hidden_dim, config.hidden_dim)

        # Final stage: ResNet blocks
        self.final_blocks = nn.ModuleList([
            ResNetBlock1D(config.hidden_dim) for _ in range(config.num_resnet_blocks)
        ])

        # Output projection to mean and logvar
        self.output_proj = CausalConv1d(config.hidden_dim, config.latent_dim * 2, kernel_size=1)

        # Initialize output projection to small values for training stability
        nn.init.zeros_(self.output_proj.conv.bias)
        nn.init.normal_(self.output_proj.conv.weight, std=0.01)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor of shape (B, T, input_dim)
        Returns:
            mean: Latent mean of shape (B, T//4, latent_dim)
            logvar: Latent log variance of shape (B, T//4, latent_dim)
        """
        # Permute to (B, C, T) for causal conv operations
        x = x.permute(0, 2, 1)

        # Initial projection (1D causal convolution)
        x = self.input_proj(x)

        # Stage 1
        for block in self.stage1_blocks:
            x = block(x)
        x = self.downsample1(x)

        # Stage 2
        for block in self.stage2_blocks:
            x = block(x)
        x = self.downsample2(x)

        # Final stage
        for block in self.final_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)

        # Permute back to (B, T//4, 2*latent_dim)
        x = x.permute(0, 2, 1)

        # Split into mean and logvar
        mean, logvar = x.chunk(2, dim=-1)

        # Clamp logvar for numerical stability (prevents exploding KL)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mean, logvar


class MotionDecoder(nn.Module):
    """Decoder for motion VAE with 4x temporal upsampling."""

    def __init__(self, config: MotionVAEConfig):
        super().__init__()
        self.config = config

        # Initial projection
        self.input_proj = CausalConv1d(config.latent_dim, config.hidden_dim, kernel_size=3)

        # Stage 1: ResNet blocks + upsample
        self.stage1_blocks = nn.ModuleList([
            ResNetBlock1D(config.hidden_dim) for _ in range(config.num_resnet_blocks)
        ])
        self.upsample1 = UpsampleBlock(config.hidden_dim, config.hidden_dim)

        # Stage 2: ResNet blocks + upsample
        self.stage2_blocks = nn.ModuleList([
            ResNetBlock1D(config.hidden_dim) for _ in range(config.num_resnet_blocks)
        ])
        self.upsample2 = UpsampleBlock(config.hidden_dim, config.hidden_dim)

        # Final stage: ResNet blocks
        self.final_blocks = nn.ModuleList([
            ResNetBlock1D(config.hidden_dim) for _ in range(config.num_resnet_blocks)
        ])

        # Output projection
        self.output_proj = CausalConv1d(config.hidden_dim, config.input_dim, kernel_size=1)

        # Initialize output projection for stability
        nn.init.zeros_(self.output_proj.conv.bias)
        nn.init.normal_(self.output_proj.conv.weight, std=0.01)

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Latent tensor of shape (B, T//4, latent_dim)
        Returns:
            Reconstructed motion of shape (B, T, input_dim)
        """
        # Permute to (B, C, T//4) for conv operations
        x = z.permute(0, 2, 1)

        # Initial projection
        x = self.input_proj(x)

        # Stage 1
        for block in self.stage1_blocks:
            x = block(x)
        x = self.upsample1(x)

        # Stage 2
        for block in self.stage2_blocks:
            x = block(x)
        x = self.upsample2(x)

        # Final stage
        for block in self.final_blocks:
            x = block(x)

        # Output projection
        x = self.output_proj(x)

        # Permute back to (B, T, input_dim)
        x = x.permute(0, 2, 1)

        return x


class MotionVAE(nn.Module):
    """Complete Motion VAE model.

    Combines preprocessing, encoder, and decoder into a single module.
    """

    def __init__(self, config: MotionVAEConfig):
        super().__init__()
        self.config = config

        self.preprocessor = NymeriaMotionPreprocessor(num_joints=config.num_joints)
        self.encoder = MotionEncoder(config)
        self.decoder = MotionDecoder(config)

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for VAE sampling.

        Args:
            mean: Mean of latent distribution (B, T//4, latent_dim)
            logvar: Log variance of latent distribution (B, T//4, latent_dim)

        Returns:
            Sampled latent vector (B, T//4, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            # During inference, just use the mean
            return mean

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode motion to latent space.

        Args:
            x: Motion tensor of shape (B, T, input_dim)

        Returns:
            mean: Latent mean (B, T//4, latent_dim)
            logvar: Latent log variance (B, T//4, latent_dim)
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to motion.

        Args:
            z: Latent tensor of shape (B, T//4, latent_dim)

        Returns:
            Reconstructed motion (B, T, input_dim)
        """
        return self.decoder(z)

    def forward(self, batch, preprocess: bool = True) -> dict:
        """Forward pass through the VAE.

        Args:
            batch: Either BatchedNymeriaTrainingSeq (if preprocess=True) or
                   preprocessed tensor of shape (B, T, input_dim)
            preprocess: Whether to preprocess the batch

        Returns:
            Dictionary containing:
                - recon: Reconstructed motion (B, T, input_dim)
                - target: Original motion (B, T, input_dim)
                - mean: Latent mean (B, T//4, latent_dim)
                - logvar: Latent log variance (B, T//4, latent_dim)
                - z: Sampled latent (B, T//4, latent_dim)
                - padding_mask: Optional padding mask (B, T)
        """
        if preprocess:
            motion = self.preprocessor(batch)
            x = motion.to_flat_tensor()
            padding_mask = motion.padding_mask
        else:
            x = batch
            padding_mask = None

        # Encode
        mean, logvar = self.encoder(x)

        # Sample
        z = self.reparameterize(mean, logvar)

        # Decode
        recon = self.decoder(z)

        return {
            'recon': recon,
            'target': x,
            'mean': mean,
            'logvar': logvar,
            'z': z,
            'padding_mask': padding_mask,
        }

    def sample(self, num_samples: int, seq_length: int, device: torch.device) -> Tensor:
        """Sample new motion sequences from the prior.

        Args:
            num_samples: Number of sequences to generate
            seq_length: Length of sequences (must be divisible by 4)
            device: Device to generate on

        Returns:
            Generated motion sequences (num_samples, seq_length, input_dim)
        """
        latent_length = seq_length // 4
        z = torch.randn(num_samples, latent_length, self.config.latent_dim, device=device)
        return self.decode(z)
