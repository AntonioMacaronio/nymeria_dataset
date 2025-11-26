"""
Nymeria dataset dataloader for ego-motion generation.

This module provides a dataloader that reads Nymeria HDF5 files from S3
(or local paths) and loads them into memory for training.

HDF5 Structure:
<sequence_name>_<datapoint_id>.h5
├── attributes: sequence_name, start_idx, end_idx, atomic_action, num_frames
├── timestamp_ns:           (N, ) array
├── root_translation:       (N, 3) array
├── root_orientation:       (N, 3, 3) array
├── cpf_translation:        (N, 3) array
├── cpf_orientation:        (N, 3, 3) array
├── joint_translation:      (N, 23, 3) array
├── joint_orientation:      (N, 23, 3, 3) array
├── contact_information:    (N, 4) array
└── egoview_RGB:            (N, 3, 1408, 1408) array

Typical Usage: 
    1. Create a dataloader with torch.util.data.DataLoader(NymeriaDataset) with batch_size > 1
    2. Use nymeria_collate_fn as the collate function (handles variable-length sequences)
    3. Loop over dataloader - it will return BatchedNymeriaTrainingSeq objects
    
Note: Variable-length sequences are automatically handled by padding (shorter sequences)
      or trimming (longer sequences) to a specified sequence_length in the collate function.
"""

from pathlib import Path
from typing import Optional
import h5py
import hdf5plugin  # Required for reading LZ4-compressed HDF5 files
import numpy as np
import torch
from torch.utils.data import Dataset


class NymeriaTrainingSeq:
    """
    A class that loads and unpacks a single Nymeria HDF5 datapoint into RAM at initialization.
    """

    def __init__(self, hdf5_path: str | Path):
        """
        Initialize and load the NymeriaTrainingSeq object into memory.

        Args:
            hdf5_path: Path to the HDF5 file (can be local or S3-mounted path)
        Raises:
            FileNotFoundError: If the HDF5 file doesn't exist
            KeyError: If expected datasets are missing from the HDF5 file
        """
        self.hdf5_path = Path(hdf5_path)

        # Validate file exists
        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        # Load all data from HDF5 into RAM
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load metadata attributes
            self.sequence_name: str = f.attrs.get('sequence_name', '')
            self.start_idx: int = f.attrs.get('start_idx', 0)
            self.end_idx: int = f.attrs.get('end_idx', 0)
            self.atomic_action: str = f.attrs.get('atomic_action', '')
            self.num_frames: int = f.attrs.get('num_frames', 0)

            # Load all data arrays into RAM
            self.timestamp_ns: np.ndarray = f['timestamp_ns'][:]                # (N,)
            self.root_translation: np.ndarray = f['root_translation'][:]        # (N, 3)
            self.root_orientation: np.ndarray = f['root_orientation'][:]        # (N, 3, 3)
            self.cpf_translation: np.ndarray = f['cpf_translation'][:]          # (N, 3)
            self.cpf_orientation: np.ndarray = f['cpf_orientation'][:]          # (N, 3, 3)
            self.joint_translation: np.ndarray = f['joint_translation'][:]      # (N, 23, 3)
            self.joint_orientation: np.ndarray = f['joint_orientation'][:]      # (N, 23, 3, 3)
            self.contact_information: np.ndarray = f['contact_information'][:]  # (N, 4)
            self.egoview_RGB: np.ndarray = f['egoview_RGB'][:]                  # (N, 3, 1408, 1408)

    @classmethod
    def from_arrays(cls,
                    timestamp_ns: np.ndarray,
                    root_translation: np.ndarray,
                    root_orientation: np.ndarray,
                    cpf_translation: np.ndarray,
                    cpf_orientation: np.ndarray,
                    joint_translation: np.ndarray,
                    joint_orientation: np.ndarray,
                    contact_information: np.ndarray,
                    egoview_RGB: np.ndarray,
                    sequence_name: str = '',
                    start_idx: int = 0,
                    end_idx: int = 0,
                    atomic_action: str = '',
                    hdf5_path: Optional[Path] = None) -> 'NymeriaTrainingSeq':
        """
        Create a NymeriaTrainingSeq object from numpy arrays directly.
        This is useful for creating sliced or padded sequences without loading from HDF5.
        
        Args:
            All the data arrays and metadata
        Returns:
            NymeriaTrainingSeq object
        """
        # Create an instance without calling __init__
        instance = cls.__new__(cls)
        
        # Set all attributes directly
        instance.timestamp_ns = timestamp_ns
        instance.root_translation = root_translation
        instance.root_orientation = root_orientation
        instance.cpf_translation = cpf_translation
        instance.cpf_orientation = cpf_orientation
        instance.joint_translation = joint_translation
        instance.joint_orientation = joint_orientation
        instance.contact_information = contact_information
        instance.egoview_RGB = egoview_RGB
        
        # Set metadata
        instance.sequence_name = sequence_name
        instance.start_idx = start_idx
        instance.end_idx = end_idx
        instance.atomic_action = atomic_action
        instance.num_frames = len(timestamp_ns)
        instance.hdf5_path = hdf5_path if hdf5_path is not None else Path('')
        
        return instance

    def get_frame_slice(self, start_frame: int, end_frame: int) -> 'NymeriaTrainingSeq':
        """
        Get a slice of frames from the loaded data.

        Args:
            start_frame: Starting frame index (inclusive)
            end_frame: Ending frame index (exclusive)
        Returns:
            NymeriaTrainingSeq object containing sliced arrays for all fields
        """
        return NymeriaTrainingSeq.from_arrays(
            timestamp_ns=self.timestamp_ns[start_frame:end_frame],
            root_translation=self.root_translation[start_frame:end_frame],
            root_orientation=self.root_orientation[start_frame:end_frame],
            cpf_translation=self.cpf_translation[start_frame:end_frame],
            cpf_orientation=self.cpf_orientation[start_frame:end_frame],
            joint_translation=self.joint_translation[start_frame:end_frame],
            joint_orientation=self.joint_orientation[start_frame:end_frame],
            contact_information=self.contact_information[start_frame:end_frame],
            egoview_RGB=self.egoview_RGB[start_frame:end_frame],
            sequence_name=self.sequence_name,
            start_idx=self.start_idx + start_frame,
            end_idx=self.start_idx + end_frame,
            atomic_action=self.atomic_action,
            hdf5_path=self.hdf5_path
        )
    
    def pad_or_trim_sequence(self, sequence_length: int = 151) -> tuple['NymeriaTrainingSeq', np.ndarray]:
        """
        Pad or trim a sequence to the given length.
        
        Args:
            sequence_length: Target sequence length (default: 151)
        Returns:
            tuple containing:
                - NymeriaTrainingSeq: the padded/trimmed sequence
                - padding_mask: np.ndarray - a mask of the same length as the sequence, 
                  with 1s for valid frames and 0s for padded frames
        """
        if len(self) >= sequence_length:  # if too long, trim the end off
            return self.get_frame_slice(0, sequence_length), np.ones(sequence_length)
        else:  # if too short, pad the end with zeros
            padding_mask = np.zeros(sequence_length)
            padding_mask[:len(self)] = 1
            
            # Calculate how many frames to pad
            num_padding_frames = sequence_length - len(self)
            
            # Create padded arrays for each field
            # For timestamp_ns (N,) - pad with zeros
            padded_timestamp_ns = np.concatenate([self.timestamp_ns,np.zeros(num_padding_frames, dtype=self.timestamp_ns.dtype)])
            
            # For root_translation (N, 3) - pad with zeros
            padded_root_translation = np.concatenate([
                self.root_translation,
                np.zeros((num_padding_frames, 3), dtype=self.root_translation.dtype)])
            
            # For root_orientation (N, 3, 3) - pad with zeros
            padded_root_orientation = np.concatenate([
                self.root_orientation,
                np.zeros((num_padding_frames, 3, 3), dtype=self.root_orientation.dtype)
            ])
            
            # For cpf_translation (N, 3) - pad with zeros
            padded_cpf_translation = np.concatenate([
                self.cpf_translation,
                np.zeros((num_padding_frames, 3), dtype=self.cpf_translation.dtype)
            ])
            
            # For cpf_orientation (N, 3, 3) - pad with zeros
            padded_cpf_orientation = np.concatenate([
                self.cpf_orientation,
                np.zeros((num_padding_frames, 3, 3), dtype=self.cpf_orientation.dtype)
            ])
            
            # For joint_translation (N, 23, 3) - pad with zeros
            padded_joint_translation = np.concatenate([
                self.joint_translation,
                np.zeros((num_padding_frames, 23, 3), dtype=self.joint_translation.dtype)
            ])
            
            # For joint_orientation (N, 23, 3, 3) - pad with zeros
            padded_joint_orientation = np.concatenate([
                self.joint_orientation,
                np.zeros((num_padding_frames, 23, 3, 3), dtype=self.joint_orientation.dtype)
            ])
            
            # For contact_information (N, 4) - pad with zeros
            padded_contact_information = np.concatenate([
                self.contact_information,
                np.zeros((num_padding_frames, 4), dtype=self.contact_information.dtype)
            ])
            
            # For egoview_RGB (N, 3, 1408, 1408) - pad with zeros
            padded_egoview_RGB = np.concatenate([
                self.egoview_RGB,
                np.zeros((num_padding_frames, 3, 1408, 1408), dtype=self.egoview_RGB.dtype)
            ])
            
            # Create new NymeriaTrainingSeq object with padded arrays
            padded_seq = NymeriaTrainingSeq.from_arrays(
                timestamp_ns=padded_timestamp_ns,
                root_translation=padded_root_translation,
                root_orientation=padded_root_orientation,
                cpf_translation=padded_cpf_translation,
                cpf_orientation=padded_cpf_orientation,
                joint_translation=padded_joint_translation,
                joint_orientation=padded_joint_orientation,
                contact_information=padded_contact_information,
                egoview_RGB=padded_egoview_RGB,
                sequence_name=self.sequence_name,
                start_idx=self.start_idx,
                end_idx=self.end_idx,
                atomic_action=self.atomic_action,
                hdf5_path=self.hdf5_path
            )
            
            return padded_seq, padding_mask


    def __repr__(self) -> str:
        return (
            f"NymeriaTrainingSeq(\n"
            f"  sequence_name='{self.sequence_name}',\n"
            f"  atomic_action='{self.atomic_action[:50]}...',\n"
            f"  num_frames={self.num_frames},\n"
            f"  hdf5_path='{self.hdf5_path}'\n"
            f")"
        )

    def __len__(self) -> int:
        """Return the number of frames in this sequence."""
        return self.num_frames


class NymeriaDataset(Dataset):
    """
    PyTorch Dataset wrapper for Nymeria HDF5 files.

    This dataset discovers all HDF5 files in a directory at initialization,
    but only loads them into RAM when accessed via __getitem__.

    Args:
        data_dir: Path to directory containing HDF5 files
        file_pattern: Glob pattern to match HDF5 files (default: "*.h5")

    Example:
        >>> dataset = NymeriaDataset("/nfs/antzhan/nymeria/hdf5")
        >>> print(f"Found {len(dataset)} datapoints")
        >>> seq = dataset[0]  # Loads first HDF5 file into RAM
    """

    def __init__(self, data_dir: str | Path, file_pattern: str = "*.h5"):
        """
        Initialize the dataset by discovering all HDF5 files.

        Args:
            data_dir: Directory containing HDF5 files
            file_pattern: Glob pattern to match files (default: "*.h5")
        Raises:
            ValueError: If directory doesn't exist or contains no HDF5 files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        if not self.data_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.data_dir}")

        # get all hdf5 paths in the directory and sort them lexicographically
        self.hdf5_paths = sorted(list(self.data_dir.glob(file_pattern)))
        if len(self.hdf5_paths) == 0:
            raise ValueError(
                f"No HDF5 files found in {self.data_dir} with pattern '{file_pattern}'"
            )
        print(f"NymeriaDataset initialized with {len(self.hdf5_paths)} HDF5 files from {self.data_dir}")

    def __len__(self) -> int:
        """Return the number of datapoints (HDF5 files) in the dataset."""
        return len(self.hdf5_paths)

    def __getitem__(self, index: int) -> NymeriaTrainingSeq:
        """
        This method loads an HDF5 file into RAM using NymeriaTrainingSeq.

        Args:
            index: Index of the datapoint to load
        Returns:
            NymeriaTrainingSeq object with all data loaded into RAM
        """
        if index < 0 or index >= len(self.hdf5_paths):
            raise IndexError(f"Index {index} out of range [0, {len(self.hdf5_paths)})")

        hdf5_path = self.hdf5_paths[index]
        return NymeriaTrainingSeq(hdf5_path)
    

class BatchedNymeriaTrainingSeq:
    """
    A class that batches a list of NymeriaTrainingSeq objects into a single object.
    All sequences are padded/trimmed to the same length before batching.
    
    Handles variable-length sequences by:
    - Padding shorter sequences with zeros (padding_mask = 0 for padded frames)
    - Trimming longer sequences to the target length
    - Creating a padding_mask to indicate valid (1.0) vs padded (0.0) frames

    Optimized version: Pre-allocates arrays and fills them directly to avoid
    expensive np.stack() operations on large arrays.
    """
    def __init__(self, batch: list[NymeriaTrainingSeq], sequence_length: int = 151):
        # Metadata
        self.sequence_length = sequence_length  # T
        self.batch_size = len(batch)  # B
        self.sequence_names = [seq.sequence_name for seq in batch]
        self.start_idx = [seq.start_idx for seq in batch]
        self.end_idx = [seq.end_idx for seq in batch]
        self.atomic_actions = [seq.atomic_action for seq in batch]

        B = self.batch_size
        T = sequence_length

        # Pre-allocate numpy arrays with correct shapes (all zeros by default)
        # This avoids creating intermediate lists and stacking
        padding_mask_np = np.zeros((B, T), dtype=np.float32)
        timestamp_ns_np = np.zeros((B, T), dtype=np.int64)
        root_translation_np = np.zeros((B, T, 3), dtype=np.float32)
        root_orientation_np = np.zeros((B, T, 3, 3), dtype=np.float32)
        cpf_translation_np = np.zeros((B, T, 3), dtype=np.float32)
        cpf_orientation_np = np.zeros((B, T, 3, 3), dtype=np.float32)
        joint_translation_np = np.zeros((B, T, 23, 3), dtype=np.float32)
        joint_orientation_np = np.zeros((B, T, 23, 3, 3), dtype=np.float32)
        contact_information_np = np.zeros((B, T, 4), dtype=np.float32)
        egoview_RGB_np = np.zeros((B, T, 3, 1408, 1408), dtype=np.uint8)

        # Fill pre-allocated arrays directly (avoids intermediate copies)
        for i, seq in enumerate(batch):
            # Handle both padding and trimming:
            # - If len(seq) < T: seq_len = len(seq), copy all frames, rest stays as zeros (padding)
            # - If len(seq) >= T: seq_len = T, copy only first T frames (trimming)
            seq_len = min(len(seq), T)

            # Mark valid frames in padding mask (1.0 = valid, 0.0 = padded)
            padding_mask_np[i, :seq_len] = 1.0

            # Copy valid frames into pre-allocated arrays
            # For sequences shorter than T, remaining frames stay as zeros (padding)
            timestamp_ns_np[i, :seq_len] = seq.timestamp_ns[:seq_len]
            root_translation_np[i, :seq_len] = seq.root_translation[:seq_len]
            root_orientation_np[i, :seq_len] = seq.root_orientation[:seq_len]
            cpf_translation_np[i, :seq_len] = seq.cpf_translation[:seq_len]
            cpf_orientation_np[i, :seq_len] = seq.cpf_orientation[:seq_len]
            joint_translation_np[i, :seq_len] = seq.joint_translation[:seq_len]
            joint_orientation_np[i, :seq_len] = seq.joint_orientation[:seq_len]
            contact_information_np[i, :seq_len] = seq.contact_information[:seq_len]
            egoview_RGB_np[i, :seq_len] = seq.egoview_RGB[:seq_len]
            # Remaining frames stay as zeros (padding)

        # Convert to torch tensors (creates views when possible, very fast)
        self.padding_mask = torch.from_numpy(padding_mask_np)                  # (B, T)
        self.timestamp_ns = torch.from_numpy(timestamp_ns_np)                  # (B, T)
        self.root_translation = torch.from_numpy(root_translation_np)          # (B, T, 3)
        self.root_orientation = torch.from_numpy(root_orientation_np)          # (B, T, 3, 3)
        self.cpf_translation = torch.from_numpy(cpf_translation_np)            # (B, T, 3)
        self.cpf_orientation = torch.from_numpy(cpf_orientation_np)            # (B, T, 3, 3)
        self.joint_translation = torch.from_numpy(joint_translation_np)        # (B, T, 23, 3)
        self.joint_orientation = torch.from_numpy(joint_orientation_np)        # (B, T, 23, 3, 3)
        self.contact_information = torch.from_numpy(contact_information_np)    # (B, T, 4)
        self.egoview_RGB = torch.from_numpy(egoview_RGB_np)                    # (B, T, 3, 1408, 1408)


def nymeria_collate_fn(batch: list[NymeriaTrainingSeq], sequence_length: int = 151) -> BatchedNymeriaTrainingSeq:
    """
    Collate a list of NymeriaTrainingSeq objects into a single BatchedNymeriaTrainingSeq object.
    
    All sequences are automatically padded (if shorter) or trimmed (if longer) to the specified
    sequence_length. The padding_mask indicates which frames are valid (1.0) vs padded (0.0).
    
    Args:
        batch: List of NymeriaTrainingSeq objects
        sequence_length: Target length for all sequences (default: 151)
    
    Returns:
        BatchedNymeriaTrainingSeq with all sequences padded/trimmed to the same length
    """
    return BatchedNymeriaTrainingSeq(batch, sequence_length=sequence_length)


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("="*80)
    print("Example 1: Load a single HDF5 file with NymeriaTrainingSeq")
    print("="*80)
    example_path = Path("/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/temp-upload-folder-test/20230607_s0_james_johnson_act0_e72nhq_00000.h5")

    if example_path.exists():
        seq = NymeriaTrainingSeq(example_path)
        print(seq)
        print(f"\nData shapes:")
        print(f"  timestamp_ns: {seq.timestamp_ns.shape}")
        print(f"  root_translation: {seq.root_translation.shape}")
        print(f"  root_orientation: {seq.root_orientation.shape}")
        print(f"  cpf_translation: {seq.cpf_translation.shape}")
        print(f"  cpf_orientation: {seq.cpf_orientation.shape}")
        print(f"  joint_translation: {seq.joint_translation.shape}")
        print(f"  joint_orientation: {seq.joint_orientation.shape}")
        print(f"  contact_information: {seq.contact_information.shape}")
        print(f"  egoview_RGB: {seq.egoview_RGB.shape}")

        # Example: Get a slice of frames
        frame_slice = seq.get_frame_slice(0, 10)
        print(f"\nSliced first 10 frames:")
        print(f"  root_translation shape: {frame_slice.root_translation.shape}")
        
        # Example: Pad or trim sequence
        padded_seq, padding_mask = seq.pad_or_trim_sequence(151)
        print(f"\nPadded/trimmed to 151 frames:")
        print(f"  Sequence length: {len(padded_seq)}")
        print(f"  Padding mask shape: {padding_mask.shape}")
        print(f"  Valid frames: {int(padding_mask.sum())}")
    else:
        print(f"Example file not found: {example_path}")

    print("\n" + "="*80)
    print("Example 2: Use NymeriaDataset with PyTorch DataLoader")
    print("="*80)

    # Example data directory (adjust to your actual path)
    data_dir = Path("/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/temp-upload-folder-test")

    if data_dir.exists():
        # Create dataset
        dataset = NymeriaDataset(data_dir)
        print(f"Dataset contains {len(dataset)} datapoints\n")

        # Create DataLoader with custom collate function
        # Note: The collate_fn automatically pads/trims all sequences to sequence_length=151
        # This handles variable-length sequences in the batch
        # Use num_workers=0 for debugging, increase for training
        sequence_length = 151  # All sequences will be padded/trimmed to this length
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=lambda batch: nymeria_collate_fn(batch, sequence_length=sequence_length),
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # Iterate through a few batches
        print("Loading first 2 batches from DataLoader:")
        for i, batch_seq in enumerate(dataloader):
            if i >= 2:  # Only show first 2 batches
                break

            # batch_seq is a BatchedNymeriaTrainingSeq object
            print(f"\nBatch at index {i}:")
            print(f"  Sequence names:           {batch_seq.sequence_names}")
            print(f"  Batch size:               {batch_seq.batch_size}")
            print(f"  Sequence length:          {batch_seq.sequence_length}")
            print(f"  Start indices:            {batch_seq.start_idx}")
            print(f"  End indices:              {batch_seq.end_idx}")
            print(f"  Atomic actions:           {batch_seq.atomic_actions}")
            print(f"  Padding mask shape:       {batch_seq.padding_mask.shape}")
            # Show valid frames per sequence (demonstrates variable-length handling)
            valid_frames = batch_seq.padding_mask.sum(dim=1).int().tolist()
            print(f"  Valid frames per seq:     {valid_frames}")
            print(f"  Timestamp ns shape:       {batch_seq.timestamp_ns.shape}")
            print(f"  Root translation shape:   {batch_seq.root_translation.shape}")
            print(f"  Root orientation shape:   {batch_seq.root_orientation.shape}")
            print(f"  CPF translation shape:    {batch_seq.cpf_translation.shape}")
            print(f"  CPF orientation shape:    {batch_seq.cpf_orientation.shape}")
            print(f"  Joint translation shape:  {batch_seq.joint_translation.shape}")
            print(f"  Joint orientation shape:  {batch_seq.joint_orientation.shape}")
            print(f"  Contact info shape:       {batch_seq.contact_information.shape}")
            print(f"  Egoview RGB shape:        {batch_seq.egoview_RGB.shape}")
    else:
        print(f"Example data directory not found: {data_dir}")
        print("To use NymeriaDataset, provide a directory path containing .h5 files")
        print("\nExample usage:")
        print("  dataset = NymeriaDataset('/nfs/antzhan/nymeria/hdf5')")
        print("  dataloader = DataLoader(dataset, batch_size=8, shuffle=True)")


