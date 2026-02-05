"""
Configuration dataclasses for Motion VAE.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MotionVAEConfig:
    """Configuration for the Motion VAE model."""

    # Architecture
    input_dim: int = 294
    """Total flattened feature dimension."""
    hidden_dim: int = 512
    """Encoder/decoder hidden channels."""
    latent_dim: int = 768
    """Latent space dimension (following EgoTwin)."""
    num_resnet_blocks: int = 2
    """Number of ResNet blocks per stage."""

    # Loss weights
    lambda_kl: float = 1e-4
    """KL divergence weight."""

    # Data dimensions
    sequence_length: int = 152
    """Must be divisible by 4 for downsampling."""
    num_joints: int = 23
    """XSens joint count."""

    # Feature dimensions (for reference)
    # hr: 6 (absolute head rotation 6D)
    # hr_dot: 6 (relative head rotation 6D)
    # hp: 3 (absolute head position)
    # hp_dot: 3 (relative head position/velocity)
    # jp: 23*3 = 69 (joint positions in head space)
    # jv: 23*3 = 69 (joint velocities in head space)
    # jr: 23*6 = 138 (joint local rotations 6D)
    # Total: 6+6+3+3+69+69+138 = 294


@dataclass
class MotionVAETrainingConfig:
    """Configuration for Motion VAE training."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # LR schedule
    warmup_steps: int = 1000

    # Training
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip: float = 1.0

    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    eval_interval: int = 500

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = True
    """Enable wandb logging."""
    project: str = "motion-vae"
    """Wandb project name."""
    entity: Optional[str] = None
    """Wandb entity (team or username). None uses the default entity."""
    run_name: Optional[str] = None
    """Wandb run name. None lets wandb auto-generate one."""
    tags: tuple[str, ...] = ()
    """Tags for this run."""


@dataclass
class TrainArgs:
    """Top-level training arguments for Motion VAE.

    Usage: python -m motion_vae.train --data-dir /path/to/hdf5
    """

    data_dir: str
    """Path to Nymeria HDF5 data directory."""

    model: MotionVAEConfig = field(default_factory=MotionVAEConfig)
    """Model architecture configuration."""
    train: MotionVAETrainingConfig = field(default_factory=MotionVAETrainingConfig)
    """Training hyperparameters."""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    """Wandb logging configuration."""

    num_workers: int = 4
    """Number of dataloader workers."""
    resume: Optional[str] = None
    """Path to checkpoint to resume from."""
    seed: int = 42
    """Random seed."""
