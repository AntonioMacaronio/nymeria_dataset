"""
Training script for Motion VAE.

Implements the EgoTwin-style grouped loss function and training loop.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import MotionVAEConfig, MotionVAETrainingConfig
from .vae_model import MotionVAE
from .motion_preprocessing import get_feature_slices

# Import Nymeria dataset components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from nymeria.nymeria_dataloader import NymeriaDataset, nymeria_collate_fn


def compute_reconstruction_loss(
    recon: Tensor,
    target: Tensor,
    padding_mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Compute grouped reconstruction loss.

    Computes MSE loss separately for each feature group following EgoTwin paper.

    Args:
        recon: Reconstructed features (B, T, 294)
        target: Target features (B, T, 294)
        padding_mask: Optional mask for valid frames (B, T)

    Returns:
        Dictionary with loss for each feature group and total loss.
    """
    feature_slices = get_feature_slices()
    losses = {}

    for name, (start, end) in feature_slices.items():
        recon_group = recon[..., start:end]
        target_group = target[..., start:end]

        # Compute MSE
        mse = (recon_group - target_group).pow(2)

        if padding_mask is not None:
            # Mask out padded frames
            mask = padding_mask.unsqueeze(-1)  # (B, T, 1)
            mse = mse * mask
            # Average over valid frames only
            loss = mse.sum() / (mask.sum() * mse.shape[-1] + 1e-8)
        else:
            loss = mse.mean()

        losses[f'rec_{name}'] = loss

    # Total reconstruction loss (average of all groups)
    losses['rec_total'] = sum(losses.values()) / len(feature_slices)

    return losses


def compute_kl_loss(
    mean: Tensor,
    logvar: Tensor,
    padding_mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """Compute KL divergence loss.

    KL divergence from N(mean, var) to N(0, 1):
    KL = -0.5 * sum(1 + log(var) - mean^2 - var)

    Args:
        mean: Latent mean (B, T//4, latent_dim)
        logvar: Latent log variance (B, T//4, latent_dim)
        padding_mask: Optional mask for valid frames (B, T)

    Returns:
        Dictionary with KL loss.
    """
    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())

    if padding_mask is not None:
        # Downsample mask to match latent temporal resolution
        B, T = padding_mask.shape
        latent_T = T // 4
        # Use max pooling to get valid latent frames (frame is valid if any of its 4 source frames were valid)
        latent_mask = padding_mask.view(B, latent_T, 4).max(dim=-1)[0]  # (B, T//4)
        latent_mask = latent_mask.unsqueeze(-1)  # (B, T//4, 1)
        kl = kl * latent_mask
        kl_loss = kl.sum() / (latent_mask.sum() * kl.shape[-1] + 1e-8)
    else:
        kl_loss = kl.mean()

    return {'kl': kl_loss}


def compute_vae_loss(
    output: dict,
    config: MotionVAEConfig,
) -> dict[str, Tensor]:
    """Compute complete VAE loss following EgoTwin paper.

    L_VAE = 1/4 * sum_c (L_rec^c + lambda_KL * L_KL^c)

    The KL loss is applied uniformly, not per feature group.

    Args:
        output: Model output dictionary
        config: Model configuration

    Returns:
        Dictionary with all loss components and total loss.
    """
    recon = output['recon']
    target = output['target']
    mean = output['mean']
    logvar = output['logvar']
    padding_mask = output.get('padding_mask')

    # Reconstruction losses (per feature group)
    rec_losses = compute_reconstruction_loss(recon, target, padding_mask)

    # KL divergence loss
    kl_losses = compute_kl_loss(mean, logvar, padding_mask)

    # Combine all losses
    losses = {**rec_losses, **kl_losses}

    # Total VAE loss: average reconstruction + weighted KL
    # Following EgoTwin: L_VAE = 1/4 * sum_c (L_rec^c) + lambda_KL * L_KL
    total_loss = rec_losses['rec_total'] + config.lambda_kl * kl_losses['kl']
    losses['total'] = total_loss

    return losses


class MotionVAETrainer:
    """Trainer class for Motion VAE."""

    def __init__(
        self,
        model: MotionVAE,
        config: MotionVAEConfig,
        train_config: MotionVAETrainingConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: torch.device = torch.device('cuda'),
    ):
        self.model = model.to(device)
        self.config = config
        self.train_config = train_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=train_config.betas,
        )

        # Learning rate scheduler with warmup
        num_training_steps = len(train_dataloader) * train_config.num_epochs
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=train_config.warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - train_config.warmup_steps,
            eta_min=train_config.learning_rate * 0.01,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[train_config.warmup_steps],
        )

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    def train_step(self, batch) -> dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        batch.cpf_translation = batch.cpf_translation.to(self.device)
        batch.cpf_orientation = batch.cpf_orientation.to(self.device)
        batch.joint_translation = batch.joint_translation.to(self.device)
        batch.joint_orientation = batch.joint_orientation.to(self.device)
        batch.padding_mask = batch.padding_mask.to(self.device)

        # Forward pass
        output = self.model(batch)

        # Compute loss
        losses = compute_vae_loss(output, self.config)

        # Backward pass
        self.optimizer.zero_grad()
        losses['total'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.train_config.gradient_clip,
        )

        self.optimizer.step()
        self.scheduler.step()

        self.global_step += 1

        return {k: v.item() for k, v in losses.items()}

    @torch.no_grad()
    def eval_step(self, batch) -> dict[str, float]:
        """Single evaluation step."""
        self.model.eval()

        # Move batch to device
        batch.cpf_translation = batch.cpf_translation.to(self.device)
        batch.cpf_orientation = batch.cpf_orientation.to(self.device)
        batch.joint_translation = batch.joint_translation.to(self.device)
        batch.joint_orientation = batch.joint_orientation.to(self.device)
        batch.padding_mask = batch.padding_mask.to(self.device)

        # Forward pass
        output = self.model(batch)

        # Compute loss
        losses = compute_vae_loss(output, self.config)

        return {k: v.item() for k, v in losses.items()}

    def validate(self) -> dict[str, float]:
        """Run validation on entire validation set."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        total_losses = {}
        num_batches = 0

        for batch in self.val_dataloader:
            losses = self.eval_step(batch)
            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1

        return {k: v / num_batches for k, v in total_losses.items()}

    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
            'train_config': self.train_config,
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.train_config.num_epochs} epochs")
        print(f"Total steps: {len(self.train_dataloader) * self.train_config.num_epochs}")

        for epoch in range(self.train_config.num_epochs):
            epoch_losses = {}
            num_batches = 0

            for batch in self.train_dataloader:
                losses = self.train_step(batch)

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v
                num_batches += 1

                # Logging
                if self.global_step % self.train_config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {self.global_step} | "
                          f"Loss: {losses['total']:.4f} | "
                          f"Rec: {losses['rec_total']:.4f} | "
                          f"KL: {losses['kl']:.6f} | "
                          f"LR: {lr:.2e}")

                # Validation
                if self.global_step % self.train_config.eval_interval == 0 and self.val_dataloader is not None:
                    val_losses = self.validate()
                    print(f"Validation | "
                          f"Loss: {val_losses['total']:.4f} | "
                          f"Rec: {val_losses['rec_total']:.4f} | "
                          f"KL: {val_losses['kl']:.6f}")

                    # Save best model
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint(
                            os.path.join(self.train_config.checkpoint_dir, 'checkpoint.pt'),
                            is_best=True,
                        )

                # Save checkpoint
                if self.global_step % self.train_config.save_interval == 0:
                    self.save_checkpoint(
                        os.path.join(self.train_config.checkpoint_dir, f'checkpoint_{self.global_step}.pt'),
                    )

            # End of epoch logging
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            print(f"Epoch {epoch+1}/{self.train_config.num_epochs} | "
                  f"Avg Loss: {avg_losses['total']:.4f} | "
                  f"Avg Rec: {avg_losses['rec_total']:.4f} | "
                  f"Avg KL: {avg_losses['kl']:.6f}")

        # Save final checkpoint
        self.save_checkpoint(
            os.path.join(self.train_config.checkpoint_dir, 'checkpoint_final.pt'),
        )
        print("Training complete!")


def create_dataloaders(
    data_dir: str,
    batch_size: int,
    sequence_length: int = 152,
    num_workers: int = 4,
    image_resolution: Optional[int] = None,
    val_split: float = 0.1,
):
    """Create training and validation dataloaders.

    Args:
        data_dir: Path to Nymeria HDF5 data directory
        batch_size: Batch size
        sequence_length: Sequence length (must be divisible by 4)
        num_workers: Number of dataloader workers
        image_resolution: Optional image resolution (None to disable video loading)
        val_split: Fraction of data to use for validation

    Returns:
        train_dataloader, val_dataloader
    """
    dataset = NymeriaDataset(data_dir, image_resolution=image_resolution)

    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    collate_fn = lambda batch: nymeria_collate_fn(batch, sequence_length=sequence_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


def main():
    parser = argparse.ArgumentParser(description='Train Motion VAE')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Nymeria HDF5 data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=152, help='Sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Configuration
    config = MotionVAEConfig(sequence_length=args.sequence_length)
    train_config = MotionVAETrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Create model
    model = MotionVAE(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers,
        image_resolution=64,  # Small resolution for faster loading (we don't use images)
    )

    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = MotionVAETrainer(
        model=model,
        config=config,
        train_config=train_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at step {trainer.global_step}")

    # Train
    trainer.train()


if __name__ == '__main__':
    main()
