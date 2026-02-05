"""
Motion VAE module for Nymeria dataset.

Implements a Continuous Motion VAE based on the EgoTwin paper (arXiv:2508.13013).
The VAE encodes motion sequences into a compressed latent space using head-centric
representation and 1D causal convolutions.
"""

from .config import MotionVAEConfig, MotionVAETrainingConfig, WandbConfig, TrainArgs
from .rotation_utils import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    compute_rotation_velocity_6d,
)
from .motion_preprocessing import (
    HeadCentricMotion,
    NymeriaMotionPreprocessor,
    get_feature_slices,
)
from .motion_dataloader import (
    MotionOnlySeq,
    MotionOnlyDataset,
    BatchedMotionSeq,
    motion_collate_fn,
)
from .vae_model import (
    CausalConv1d,
    ResNetBlock1D,
    DownsampleBlock,
    UpsampleBlock,
    MotionEncoder,
    MotionDecoder,
    MotionVAE,
)

# Training utilities - imported lazily to avoid dependency issues
# Use: from motion_vae.train import compute_vae_loss, MotionVAETrainer, etc.
def _lazy_import_train():
    from .train import (
        compute_reconstruction_loss,
        compute_kl_loss,
        compute_vae_loss,
        MotionVAETrainer,
        create_dataloaders,
    )
    return {
        'compute_reconstruction_loss': compute_reconstruction_loss,
        'compute_kl_loss': compute_kl_loss,
        'compute_vae_loss': compute_vae_loss,
        'MotionVAETrainer': MotionVAETrainer,
        'create_dataloaders': create_dataloaders,
    }

__all__ = [
    # Config
    'MotionVAEConfig',
    'MotionVAETrainingConfig',
    # Rotation utilities
    'matrix_to_rotation_6d',
    'rotation_6d_to_matrix',
    'compute_rotation_velocity_6d',
    # Preprocessing
    'HeadCentricMotion',
    'NymeriaMotionPreprocessor',
    'get_feature_slices',
    # Model components
    'CausalConv1d',
    'ResNetBlock1D',
    'DownsampleBlock',
    'UpsampleBlock',
    'MotionEncoder',
    'MotionDecoder',
    'MotionVAE',
    # Training (import from motion_vae.train directly)
    # 'compute_reconstruction_loss',
    # 'compute_kl_loss',
    # 'compute_vae_loss',
    # 'MotionVAETrainer',
    # 'create_dataloaders',
]
