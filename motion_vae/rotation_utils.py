"""
Rotation utilities for 6D rotation representation.

Based on "On the Continuity of Rotation Representations in Neural Networks" (Zhou et al., 2019).
The 6D representation uses the first two columns of the rotation matrix, which provides
a continuous representation suitable for neural network learning.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def matrix_to_rotation_6d(R: Tensor) -> Tensor:
    """Convert rotation matrices to 6D representation.

    The 6D representation consists of the first two columns of the rotation matrix
    flattened into a 6D vector. This provides a continuous representation of rotations.

    Args:
        R: Rotation matrices of shape (..., 3, 3)

    Returns:
        6D rotation representation of shape (..., 6)
    """
    # Take first two columns and flatten
    # R[..., :, :2] has shape (..., 3, 2)
    # We need to transpose to get (..., 2, 3) then flatten to (..., 6)
    # Or equivalently, flatten the first two columns directly
    batch_shape = R.shape[:-2]
    return R[..., :, :2].transpose(-2, -1).reshape(*batch_shape, 6)


def rotation_6d_to_matrix(rot6d: Tensor) -> Tensor:
    """Convert 6D rotation representation back to rotation matrices via Gram-Schmidt.

    Args:
        rot6d: 6D rotation representation of shape (..., 6)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    # Split into two 3D vectors (the two columns of the rotation matrix)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=-1)
    # Remove component of a2 parallel to b1
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    # Third column is cross product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack to form rotation matrix
    # b1, b2, b3 are columns, so we stack and transpose
    return torch.stack([b1, b2, b3], dim=-1)


def compute_rotation_velocity_6d(R: Tensor, R_prev: Tensor) -> Tensor:
    """Compute relative rotation between consecutive frames in 6D representation.

    Computes R_prev^T @ R which represents the rotation from frame t-1 to frame t,
    then converts to 6D representation.

    Args:
        R: Current rotation matrices of shape (..., 3, 3)
        R_prev: Previous rotation matrices of shape (..., 3, 3)

    Returns:
        Relative rotation in 6D representation of shape (..., 6)
    """
    # R_rel = R_prev^T @ R
    R_rel = torch.matmul(R_prev.transpose(-2, -1), R)
    return matrix_to_rotation_6d(R_rel)
