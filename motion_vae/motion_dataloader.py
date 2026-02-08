"""
Lightweight motion-only dataloader for VAE training.

Skips video decoding entirely — reads only HDF5 motion arrays.
This is dramatically faster than the full NymeriaDataset which decodes
MP4 video on every __getitem__ call.
"""

from pathlib import Path
from typing import Optional

import h5py
import hdf5plugin  # Required for reading LZ4-compressed HDF5 files
import numpy as np
import torch
from torch.utils.data import Dataset


class MotionOnlySeq:
    """A single Nymeria motion sequence loaded from HDF5 (no video)."""

    __slots__ = [
        'sequence_name', 'atomic_action', 'num_frames',
        'cpf_translation', 'cpf_orientation',
        'joint_translation', 'joint_orientation',
    ]

    def __init__(
        self,
        sequence_name: str,
        atomic_action: str,
        num_frames: int,
        cpf_translation: np.ndarray,
        cpf_orientation: np.ndarray,
        joint_translation: np.ndarray,
        joint_orientation: np.ndarray,
    ):
        self.sequence_name = sequence_name
        self.atomic_action = atomic_action
        self.num_frames = num_frames
        self.cpf_translation = cpf_translation      # (N, 3)
        self.cpf_orientation = cpf_orientation      # (N, 3, 3)
        self.joint_translation = joint_translation  # (N, 23, 3)
        self.joint_orientation = joint_orientation  # (N, 23, 3, 3)

    def __len__(self) -> int:
        return self.num_frames

    @staticmethod
    def from_hdf5(hdf5_path: Path) -> 'MotionOnlySeq':
        with h5py.File(hdf5_path, 'r') as f:
            return MotionOnlySeq(
                sequence_name=f.attrs.get('sequence_name', ''),
                atomic_action=f.attrs.get('atomic_action', ''),
                num_frames=int(f.attrs.get('num_frames', 0)),
                cpf_translation=f['cpf_translation'][:].astype(np.float32),
                cpf_orientation=f['cpf_orientation'][:].astype(np.float32),
                joint_translation=f['joint_translation'][:].astype(np.float32),
                joint_orientation=f['joint_orientation'][:].astype(np.float32),
            )


class MotionOnlyDataset(Dataset):
    """Dataset that loads only motion data from Nymeria HDF5 files (no video).

    Compared to NymeriaDataset, this skips MP4 video decoding, which is the
    dominant cost per sample. Only the motion arrays needed by the VAE are loaded.

    Args:
        data_dir: Path to directory containing HDF5 files
        file_pattern: Glob pattern to match HDF5 files (default: "*.h5")
    """

    def __init__(self, data_dir: str | Path, file_pattern: str = "*.h5"):
        self.data_dir = Path(data_dir)

        if not self.data_dir.is_dir():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        hdf5_paths_list = sorted(self.data_dir.glob(file_pattern))
        if len(hdf5_paths_list) == 0:
            raise ValueError(
                f"No HDF5 files found in {self.data_dir} with pattern '{file_pattern}'"
            )
        # Store as numpy string array to avoid DataLoader memory leaks
        self.hdf5_paths = np.array([str(p) for p in hdf5_paths_list], dtype=np.str_)
        print(f"MotionOnlyDataset: {len(self.hdf5_paths)} files from {self.data_dir}")

    def __len__(self) -> int:
        return len(self.hdf5_paths)

    def __getitem__(self, index: int) -> MotionOnlySeq:
        return MotionOnlySeq.from_hdf5(Path(self.hdf5_paths[index]))


class BatchedMotionSeq:
    """Batched motion-only sequences, ready for the VAE.

    Only contains the fields the VAE preprocessor needs:
    cpf_translation, cpf_orientation, joint_translation, joint_orientation, padding_mask.
    """

    __slots__ = [
        'batch_size', 'sequence_length',
        'padding_mask', 'cpf_translation', 'cpf_orientation',
        'joint_translation', 'joint_orientation',
    ]

    def __init__(self, batch: list[MotionOnlySeq], sequence_length: int = 152):
        B = len(batch)
        T = sequence_length

        padding_mask_np = np.zeros((B, T), dtype=np.float32)
        cpf_translation_np = np.zeros((B, T, 3), dtype=np.float32)
        cpf_orientation_np = np.zeros((B, T, 3, 3), dtype=np.float32)
        joint_translation_np = np.zeros((B, T, 23, 3), dtype=np.float32)
        joint_orientation_np = np.zeros((B, T, 23, 3, 3), dtype=np.float32)

        for i, seq in enumerate(batch):
            n = min(len(seq), T)
            padding_mask_np[i, :n] = 1.0
            cpf_translation_np[i, :n] = seq.cpf_translation[:n]
            cpf_orientation_np[i, :n] = seq.cpf_orientation[:n]
            joint_translation_np[i, :n] = seq.joint_translation[:n]
            joint_orientation_np[i, :n] = seq.joint_orientation[:n]

        self.batch_size = B
        self.sequence_length = T
        self.padding_mask = torch.from_numpy(padding_mask_np)
        self.cpf_translation = torch.from_numpy(cpf_translation_np)
        self.cpf_orientation = torch.from_numpy(cpf_orientation_np)
        self.joint_translation = torch.from_numpy(joint_translation_np)
        self.joint_orientation = torch.from_numpy(joint_orientation_np)

    def to(self, device: torch.device) -> 'BatchedMotionSeq':
        """Move all tensors to the given device. Returns self for chaining."""
        self.padding_mask = self.padding_mask.to(device)
        self.cpf_translation = self.cpf_translation.to(device)
        self.cpf_orientation = self.cpf_orientation.to(device)
        self.joint_translation = self.joint_translation.to(device)
        self.joint_orientation = self.joint_orientation.to(device)
        return self


def motion_collate_fn(
    batch: list[MotionOnlySeq],
    sequence_length: int = 152,
) -> BatchedMotionSeq:
    """Collate motion-only sequences into a batch."""
    return BatchedMotionSeq(batch, sequence_length=sequence_length)
