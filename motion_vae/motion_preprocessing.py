"""
Motion preprocessing for converting Nymeria data to head-centric representation.

This module transforms world-coordinate motion data from Nymeria into the head-centric
representation used by the EgoTwin paper, suitable for VAE training.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .rotation_utils import matrix_to_rotation_6d, compute_rotation_velocity_6d


@dataclass
class HeadCentricMotion:
    """Head-centric motion representation following EgoTwin paper.

    All tensors have shape (B, T, ...) where B is batch size and T is sequence length.

    Attributes:
        hr: Absolute head rotation in 6D representation                 (B, T, 6) - NOTE: absolute means in the world frame
        hr_dot: Relative head rotation (velocity) in 6D representation  (B, T, 6)
        hp: Absolute head position                                      (B, T, 3)
        hp_dot: Relative head position (velocity)                       (B, T, 3)
        jp: Joint positions in head-centric space                       (B, T, 23, 3)
        jv: Joint velocities in head-centric space                      (B, T, 23, 3)
        jr: Joint local rotations in 6D representation                  (B, T, 23, 6)
        padding_mask: Optional mask indicating valid frames             (B, T)
    """
    hr: Tensor      # (B, T, 6)
    hr_dot: Tensor  # (B, T, 6)
    hp: Tensor      # (B, T, 3)
    hp_dot: Tensor  # (B, T, 3)
    jp: Tensor      # (B, T, 23, 3)
    jv: Tensor      # (B, T, 23, 3)
    jr: Tensor      # (B, T, 23, 6)
    padding_mask: Optional[Tensor] = None  # (B, T)

    def to_flat_tensor(self) -> Tensor:
        """Flatten all features into a single tensor for VAE input.

        Returns:
            Tensor of shape (B, T, 294) containing all features concatenated.
        """
        B, T = self.hr.shape[:2]

        # Flatten spatial dimensions
        jp_flat = self.jp.reshape(B, T, -1)  # (B, T, 69)
        jv_flat = self.jv.reshape(B, T, -1)  # (B, T, 69)
        jr_flat = self.jr.reshape(B, T, -1)  # (B, T, 138)

        # Concatenate all features
        # Order: hr(6), hr_dot(6), hp(3), hp_dot(3), jp(69), jv(69), jr(138) = 294
        return torch.cat([
            self.hr,      # 6
            self.hr_dot,  # 6
            self.hp,      # 3
            self.hp_dot,  # 3
            jp_flat,      # 69
            jv_flat,      # 69
            jr_flat,      # 138
        ], dim=-1)

    @staticmethod
    def from_flat_tensor(x: Tensor, num_joints: int = 23) -> 'HeadCentricMotion':
        """Reconstruct HeadCentricMotion from flattened tensor.

        Args:
            x: Flattened tensor of shape (B, T, 294)
            num_joints: Number of joints (default 23 for XSens)

        Returns:
            HeadCentricMotion instance
        """
        B, T, _ = x.shape

        # Split features according to their dimensions
        idx = 0
        hr = x[..., idx:idx+6]
        idx += 6
        hr_dot = x[..., idx:idx+6]
        idx += 6
        hp = x[..., idx:idx+3]
        idx += 3
        hp_dot = x[..., idx:idx+3]
        idx += 3
        jp = x[..., idx:idx+num_joints*3].reshape(B, T, num_joints, 3)
        idx += num_joints * 3
        jv = x[..., idx:idx+num_joints*3].reshape(B, T, num_joints, 3)
        idx += num_joints * 3
        jr = x[..., idx:idx+num_joints*6].reshape(B, T, num_joints, 6)

        return HeadCentricMotion(
            hr=hr,
            hr_dot=hr_dot,
            hp=hp,
            hp_dot=hp_dot,
            jp=jp,
            jv=jv,
            jr=jr,
        )

    def to(self, device: torch.device) -> 'HeadCentricMotion':
        """Move all tensors to specified device."""
        return HeadCentricMotion(
            hr=self.hr.to(device),
            hr_dot=self.hr_dot.to(device),
            hp=self.hp.to(device),
            hp_dot=self.hp_dot.to(device),
            jp=self.jp.to(device),
            jv=self.jv.to(device),
            jr=self.jr.to(device),
            padding_mask=self.padding_mask.to(device) if self.padding_mask is not None else None,
        )


class NymeriaMotionPreprocessor:
    """Preprocessor to convert Nymeria batch data to head-centric representation."""

    # Head joint index in XSens skeleton
    HEAD_JOINT_IDX = 6

    def __init__(self, num_joints: int = 23):
        """
        Args:
            num_joints: Number of joints in the skeleton (default 23 for XSens)
        """
        self.num_joints = num_joints

    def __call__(self, batch) -> HeadCentricMotion:
        """Convert BatchedNymeriaTrainingSeq to HeadCentricMotion.

        Args:
            batch: BatchedNymeriaTrainingSeq with attributes:
                - cpf_translation: (B, T, 3) head position
                - cpf_orientation: (B, T, 3, 3) head rotation matrix
                - joint_translation: (B, T, 23, 3) joint positions
                - joint_orientation: (B, T, 23, 3, 3) joint rotations
                - padding_mask: (B, T) valid frame mask

        Returns:
            HeadCentricMotion instance
        """
        # Extract tensors from batch
        cpf_translation = batch.cpf_translation     # (B, T, 3) head position
        cpf_orientation = batch.cpf_orientation     # (B, T, 3, 3)
        joint_translation = batch.joint_translation # (B, T, 23, 3)
        joint_orientation = batch.joint_orientation # (B, T, 23, 3, 3)
        padding_mask = batch.padding_mask           # (B, T)

        B, T = cpf_translation.shape[:2]

        # 1. Head rotation (absolute) - convert to 6D
        hr = matrix_to_rotation_6d(cpf_orientation)  # (B, T, 6)

        # 2. Head rotation velocity (relative between frames)
        # For first frame, use identity rotation (6D: [1,0,0, 0,1,0]) because there is no previous frame!
        hr_dot = torch.zeros_like(hr)
        # Compute relative rotation for frames 1 to T-1
        hr_dot[:, 1:] = compute_rotation_velocity_6d(
            cpf_orientation[:, 1:],  # Current
            cpf_orientation[:, :-1]  # Previous
        )
        # First frame: identity rotation in 6D (first two columns of identity matrix)
        hr_dot[:, 0, 0] = 1.0  # First column: [1, 0, 0]
        hr_dot[:, 0, 3] = 1.0  # Second column: [0, 1, 0] -> but stored as [0, 1, 0] at indices 3,4,5
        hr_dot[:, 0, 4] = 1.0  # Actually: 6D is [r1_x, r1_y, r1_z, r2_x, r2_y, r2_z]
        # Identity rotation: r1=[1,0,0], r2=[0,1,0] -> 6D=[1,0,0,0,1,0]
        hr_dot[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                     device=hr_dot.device, dtype=hr_dot.dtype)

        # 3. Head position (absolute)
        hp = cpf_translation  # (B, T, 3)

        # 4. Head position velocity
        hp_dot = torch.zeros_like(hp)
        hp_dot[:, 1:] = cpf_translation[:, 1:] - cpf_translation[:, :-1]

        # 5. Joint positions in head-centric space
        # Transform world joint positions to head coordinate frame
        # p_head = R_head^T @ (p_world - t_head)
        R_head_inv = cpf_orientation.transpose(-2, -1)  # (B, T, 3, 3)
        t_head = cpf_translation.unsqueeze(2)  # (B, T, 1, 3)

        # Subtract head position and rotate to head frame
        jp_world_centered = joint_translation - t_head  # (B, T, 23, 3)
        # Apply rotation: (B, T, 3, 3) @ (B, T, 23, 3) -> need einsum
        jp = torch.einsum('btij,btnj->btni', R_head_inv, jp_world_centered)  # (B, T, 23, 3)

        # 6. Joint velocities in head-centric space
        jv = torch.zeros_like(jp)
        jv[:, 1:] = jp[:, 1:] - jp[:, :-1]

        # 7. Joint local rotations in 6D
        # Convert joint rotations to 6D representation
        jr = matrix_to_rotation_6d(joint_orientation)  # (B, T, 23, 6)

        return HeadCentricMotion(
            hr=hr,
            hr_dot=hr_dot,
            hp=hp,
            hp_dot=hp_dot,
            jp=jp,
            jv=jv,
            jr=jr,
            padding_mask=padding_mask,
        )


    def process_tensors(
        self,
        cpf_translation: Tensor,
        cpf_orientation: Tensor,
        joint_translation: Tensor,
        joint_orientation: Tensor,
        padding_mask: Tensor = None,
    ) -> HeadCentricMotion:
        """Convert raw tensors to HeadCentricMotion (same logic as __call__ but with explicit tensors).

        Args:
            cpf_translation: (B, T, 3) head position
            cpf_orientation: (B, T, 3, 3) head rotation matrix
            joint_translation: (B, T, 23, 3) joint positions
            joint_orientation: (B, T, 23, 3, 3) joint rotations
            padding_mask: (B, T) valid frame mask (optional)

        Returns:
            HeadCentricMotion instance
        """

        class _TensorHolder:
            pass

        holder = _TensorHolder()
        holder.cpf_translation = cpf_translation
        holder.cpf_orientation = cpf_orientation
        holder.joint_translation = joint_translation
        holder.joint_orientation = joint_orientation
        holder.padding_mask = padding_mask if padding_mask is not None else torch.ones(
            cpf_translation.shape[:2], device=cpf_translation.device, dtype=cpf_translation.dtype
        )
        return self(holder)


def get_feature_slices() -> dict:
    """Get feature slice indices for grouped loss computation.

    Returns:
        Dictionary mapping feature group names to (start, end) index tuples.
    """
    return {
        'head_6D': (0, 12),      # hr(6) + hr_dot(6)
        'head_3D': (12, 18),     # hp(3) + hp_dot(3)
        'joint_3D': (18, 156),   # jp(69) + jv(69)
        'joint_6D': (156, 294),  # jr(138)
    }
