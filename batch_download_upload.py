import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import tyro
import h5py
import numpy as np
from extract_antego_data import extract_to_hdf5_chunked
import shutil


WORKSPACE_ROOT = "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset"
DEFAULT_URL_JSON = f"{WORKSPACE_ROOT}/Nymeria_download_urls.json"
DEFAULT_OUT_DIR = f"{WORKSPACE_ROOT}/temp-upload-folder"
DEFAULT_S3_PREFIX = "s3://far-research-internal/antzhan/nymeria/hdf5"
DEFAULT_AWS_PROFILE = "far-compute"
DEFAULT_AWS_REGION = "us-west-2"


def load_first_n_sequence_keys(url_json_path: str, limit: int, only_with_narrations: bool = True) -> List[str]:
    with open(url_json_path, "r") as f:
        data = json.load(f)
    sequences = data.get("sequences", {})

    if only_with_narrations:
        # Filter for sequences that have narration/language annotation fields
        sequences_with_narrations = []
        for seq_key, seq_data in sequences.items():
            narration_keys = [k for k in seq_data.keys() if 'narration' in k.lower()]
            if narration_keys:
                sequences_with_narrations.append(seq_key)
        keys = sorted(sequences_with_narrations)
        print(f"Filtered to {len(keys)} sequences with language annotations (out of {len(sequences)} total)")
    else:
        # If not filtering for sequences with language annotations, return all sequences
        # NOTE: There are 1100 total sequences in the dataset, but only 864 of them have language annotations.
        keys = sorted(sequences.keys()) # keys is a length 1100 list of sequence names: ['20230607_s0_james_johnson_act0_e72nhq', '20230607_s0_james_johnson_act1_7xwm28', ...]

    return keys[:limit]


def run_download(url_json_path: str, output_dir: str, sequence_key: str) -> None:
    cmd = [
        sys.executable,
        f"{WORKSPACE_ROOT}/download.py",
        "-i",
        url_json_path,
        "-o",
        output_dir,
        "-k",
        sequence_key,
    ]
    subprocess.run(cmd, check=True, input="y\n", text=True)


def run_s3_upload(local_path: str, s3_prefix: str, aws_profile: str, aws_region: str) -> None:
    cmd = [
        "aws",
        "s3",
        "cp",
        "--recursive",
        local_path,
        f"{s3_prefix.rstrip('/')}/{Path(local_path).name}/",
        "--region",
        aws_region,
    ]
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile
    subprocess.run(cmd, check=True, env=env)


def run_s3_upload_file(local_file: str, s3_path: str, aws_profile: str, aws_region: str) -> None:
    """Upload a single file to S3"""
    cmd = [
        "aws",
        "s3",
        "cp",
        local_file,
        s3_path,
        "--region",
        aws_region,
    ]
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile
    subprocess.run(cmd, check=True, env=env)


def save_to_hdf5(data: Dict[str, Any], output_path: str) -> None:
    """Save extract_antego_data output to HDF5 file"""
    with h5py.File(output_path, 'w') as f:
        # Save scalar/array data
        for key in ['timestamp_ns', 'root_translation', 'root_orientation',
                    'cpf_translation', 'cpf_orientation',
                    'joint_translation', 'joint_orientation',
                    'contact_information', 'egoview_RGB']:
            if key in data and data[key] is not None and len(data[key]) > 0:
                # Convert list of arrays to numpy array
                arr = np.array(data[key])
                f.create_dataset(key, data=arr, compression='gzip', compression_opts=4)

        # Save pointcloud (static for entire sequence)
        if data.get('pointcloud') is not None:
            f.create_dataset('pointcloud', data=data['pointcloud'], compression='gzip', compression_opts=4)

        # Save narration DataFrames as separate groups
        for narration_key in ['motion_narration', 'activity_summarization', 'atomic_action']:
            if data.get(narration_key) is not None:
                grp = f.create_group(narration_key)
                df = data[narration_key]
                # Save each column
                for col in df.columns:
                    col_data = df[col].values
                    # Handle string columns
                    if col_data.dtype == object:
                        col_data = col_data.astype(str)
                        grp.create_dataset(col, data=col_data, dtype=h5py.string_dtype())
                    else:
                        grp.create_dataset(col, data=col_data)

    print(f"Saved HDF5 file: {output_path} ({os.path.getsize(output_path) / 1e6:.2f} MB)")



@dataclass
class DownloadUploadArgs:
    url_json: str = DEFAULT_URL_JSON    # "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/Nymeria_download_urls.json"
    """Path to `Nymeria_download_urls.json` file"""

    out_dir: str = DEFAULT_OUT_DIR      # "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/temp-upload-folder"
    """Local output directory for temporary downloads"""

    limit: int = -1
    """Number of sequences to process (alphabetically). If -1, process all sequences."""

    only_with_narrations: bool = True
    """If True, only process sequences with language annotations (narrations). 867 out of 1100 sequences have narrations."""

    s3_prefix: str = DEFAULT_S3_PREFIX  # "s3://far-research-internal/antzhan/nymeria/hdf5"
    """Destination S3 prefix for HDF5 files"""

    aws_profile: str = DEFAULT_AWS_PROFILE
    """AWS profile name to use"""

    aws_region: str = DEFAULT_AWS_REGION
    """AWS region for S3 operations"""

    # Extract parameters
    frame_rate: float = 30.0
    """Frame rate for extraction (fps)"""

    start_frame: int = 0
    """Start frame index"""

    num_frames: int = -1
    """Number of frames to extract per sequence. If -1, extract all frames in the sequence."""

    sample_rate: int = 1
    """Sample rate (1 = every frame, 2 = every other frame, etc.)"""


def main(args: DownloadUploadArgs) -> None:
    url_json_path = os.path.abspath(args.url_json)      # "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/Nymeria_download_urls.json"
    out_dir = os.path.abspath(args.out_dir)             # "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/temp-upload-folder"
    s3_prefix = args.s3_prefix                          # "s3://far-research-internal/antzhan/nymeria/hdf5"
    aws_profile = args.aws_profile                      # "far-compute"
    aws_region = args.aws_region                        # "us-west-2"
    limit = args.limit                                  # -1
    only_with_narrations = args.only_with_narrations    # True

    os.makedirs(out_dir, exist_ok=True) # Creates the temporary folder to hold hdf5 files for uploading to s3 if it doesn't exist.
    keys = load_first_n_sequence_keys(url_json_path, limit, only_with_narrations) # alphabetically sorted list of sequence keys to process.
    if not keys:
        print("No sequences found in the provided JSON.")
        sys.exit(1)

    print(f"Found {len(keys)} sequences to process.")
    print(f"Temp directory: {out_dir}")
    print(f"S3 destination: {s3_prefix}")
    print(f"Extraction params: {args.num_frames if args.num_frames != -1 else 'all'} frames @ {args.frame_rate}fps, sample_rate={args.sample_rate}\n")

    for idx, key in enumerate(keys, start=1):
        local_seq_dir = os.path.join(out_dir, key)                  # Ex: '/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq'

        print(f"\n{'='*80}")
        print(f"[{idx}/{len(keys)}] Processing: {key}")
        print(f"{'='*80}")

        ########################################################
        ####  Step 1: Download sequence  ######
        ########################################################
        print(f"[{idx}/{len(keys)}] Step 1/4: Downloading sequence...")
        try:
            run_download(url_json_path, out_dir, key)
        except subprocess.CalledProcessError as e:
            print(f"❌ Download failed for {key}: {e}")
            continue

        if not os.path.isdir(local_seq_dir):
            print(f"❌ Expected directory not found: {local_seq_dir}")
            try:
                contents = os.listdir(out_dir)
                print(f"Current contents of {out_dir}: {contents[:20]}")
            except Exception:
                pass
            continue

        ########################################################
        ####  Step 2: Process sequence into hdf5 files  ######
        ########################################################
        print(f"[{idx}/{len(keys)}] Step 2/4: Processing sequence into hdf5 files (each file is a datapoint)...")
        try:
            hdf5_file_paths = extract_to_hdf5_chunked(
                sequence_folder=Path(local_seq_dir),
                output_dir=out_dir,
                frame_rate=args.frame_rate,
            )
            print(f"✓ Processed sequence into {len(hdf5_file_paths)} hdf5 files")
        except Exception as e:
            print(f"❌ Extraction failed for {key}: {e}")
            # Clean up downloaded sequence
            if os.path.isdir(local_seq_dir):
                shutil.rmtree(local_seq_dir)
            continue

        ########################################################
        ####  Step 3: Upload HDF5 files to S3  ######
        ########################################################
        print(f"[{idx}/{len(keys)}] Step 3/4: Uploading HDF5 files to S3...")
        for hdf5_path in hdf5_file_paths:
            s3_hdf5_path = f"{s3_prefix.rstrip('/')}/{Path(hdf5_path).name}"
            try:
                run_s3_upload_file(hdf5_path, s3_hdf5_path, aws_profile, aws_region)
                print(f"✓ Uploaded to {s3_hdf5_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ S3 upload failed for {key}: {e}")
                continue

        ########################################################
        ####  Step 4: Clean up local files  ######
        ########################################################
        print(f"[{idx}/{len(keys)}] Step 4/4: Cleaning up local files...")
        try:
            if os.path.isdir(local_seq_dir):
                shutil.rmtree(local_seq_dir)
                print(f"✓ Deleted sequence directory: {local_seq_dir}")
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)
                print(f"✓ Deleted HDF5 file: {hdf5_path}")
        except Exception as e:
            print(f"⚠️  Cleanup warning for {key}: {e}")
        # also delete data_summary.json and download_summary.json if they exist
        

        print(f"✅ [{idx}/{len(keys)}] Successfully processed: {key}")

    print(f"\n{'='*80}")
    print(f"Processing complete! Processed {len(keys)} sequences.")
    print(f"{'='*80}")


if __name__ == "__main__":
    tyro.cli(main)


