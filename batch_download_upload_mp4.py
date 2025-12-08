import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List
from dataclasses import dataclass
import tyro
from extract_antego_data import extract_to_mp4_chunked
import shutil
import re


WORKSPACE_ROOT = "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset"
DEFAULT_URL_JSON = f"{WORKSPACE_ROOT}/Nymeria_download_urls.json"
DEFAULT_OUT_DIR = f"{WORKSPACE_ROOT}/temp-upload-folder"
DEFAULT_S3_PREFIX = "s3://far-research-internal/antzhan/nymeria/mp4"
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

    return keys if limit == -1 else                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            keys[:limit]


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


def get_existing_sequence_keys_from_s3(s3_prefix: str, aws_profile: str, aws_region: str) -> set:
    """List all files in S3 bucket and extract unique sequence keys.
    
    Files in S3 are named like: {sequence_key}_{chunk_number}.h5 or {sequence_key}_{chunk_number}.mp4
    For example: 20230607_s0_james_johnson_act0_e72nhq_00000.h5
    
    This function extracts the sequence keys by removing the extension and numbered suffix.
    """
    
    cmd = ["aws", "s3", "ls", s3_prefix.rstrip('/') + '/', "--region", aws_region] # ex: AWS_PROFILE=far-compute aws s3 ls s3://far-research-internal/antzhan/nymeria/mp4/ --region us-west-2
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        lines = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        # If the bucket/prefix doesn't exist or is empty, return empty set
        return set()
    
    sequence_keys = set()
    for line in lines:
        if not line.strip():
            continue
        # S3 ls output format: "2024-01-01 12:00:00    12345 filename.ext"
        parts = line.split()
        if len(parts) >= 4:
            filename = parts[-1]  # Get the filename (last part)
            # Remove extension (.h5 or .mp4)
            base_name = re.sub(r'\.(h5|mp4)$', '', filename)
            # Remove numbered suffix (e.g., _00000, _00001, etc.)
            sequence_key = re.sub(r'_\d+$', '', base_name)
            if sequence_key:
                sequence_keys.add(sequence_key)
    
    return sequence_keys


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


@dataclass
class DownloadUploadArgs:
    url_json: str = DEFAULT_URL_JSON    # "/home/ubuntu/sky_workdir/nymeria_dataset/Nymeria_download_urls.json"
    """Path to `Nymeria_download_urls.json` file"""

    out_dir: str = DEFAULT_OUT_DIR      # "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    """Local output directory for temporary downloads"""

    limit: int = -1
    """Number of sequences to process (alphabetically). If -1, process all sequences."""

    only_with_narrations: bool = True
    """If True, only process sequences with language annotations (narrations). 867 out of 1100 sequences have narrations."""

    s3_prefix: str = DEFAULT_S3_PREFIX  # "s3://far-research-internal/antzhan/nymeria/mp4   "
    """Destination S3 prefix for HDF5 and MP4 files"""

    aws_profile: str = DEFAULT_AWS_PROFILE
    """AWS profile name to use"""

    aws_region: str = DEFAULT_AWS_REGION
    """AWS region for S3 operations"""

    # Extract parameters
    frame_rate: float = 30.0
    """Frame rate for extraction (fps)"""

    resolution: int = 1408
    """Resolution for extraction (pixels)"""


def main(args: DownloadUploadArgs) -> None:
    """This script downloads a sequence from the Nymeria dataset, processes it into hdf5 and mp4 files, and uploads them to S3.
    Video frames are stored in MP4 files for compression efficiency. Other data (poses, joints, etc.) are stored in HDF5 files.
    It then cleans up the local files. It processes one sequence at a time and each sequence can have many datapoint files.
    """
    url_json_path = os.path.abspath(args.url_json)      # "/home/ubuntu/sky_workdir/nymeria_dataset/Nymeria_download_urls.json"
    out_dir = os.path.abspath(args.out_dir)             # "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    s3_prefix = args.s3_prefix                          # "s3://far-research-internal/antzhan/nymeria/mp4"
    aws_profile = args.aws_profile                      # "far-compute"
    aws_region = args.aws_region                        # "us-west-2"
    limit = args.limit                                  # -1
    only_with_narrations = args.only_with_narrations    # True

    os.makedirs(out_dir, exist_ok=True) # Creates the temporary folder to hold hdf5 and mp4 files for uploading to s3 if it doesn't exist.
    keys = load_first_n_sequence_keys(url_json_path, limit, only_with_narrations) # alphabetically sorted list of sequence keys to process.
    if not keys:
        print("No sequences found in the provided JSON.")
        sys.exit(1)

    print(f"Found {len(keys)} sequences to process.")
    print(f"Temp directory: {out_dir}")
    print(f"S3 destination: {s3_prefix}")
    print(f"Extraction params: all datapoints @ {args.frame_rate}fps\n")

    # Get existing sequence keys from S3 to skip already-uploaded sequences
    print("Checking S3 for existing sequences...")
    existing_keys = get_existing_sequence_keys_from_s3(s3_prefix, aws_profile, aws_region)
    print(f"Found {len(existing_keys)} existing sequences in S3.\n")

    for idx, key in enumerate(keys, start=1):
        local_seq_dir = os.path.join(out_dir, key)                  # Ex: '/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq'
        
        # Skip if the sequence is already in the S3 bucket
        if key in existing_keys:
            print(f"[{idx}/{len(keys)}] ⏭️  Skipping (already in S3): {key}")
            continue

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
        ####  Step 2: Process sequence into hdf5 and mp4 files  ######
        ########################################################
        print(f"[{idx}/{len(keys)}] Step 2/4: Processing sequence into hdf5 and mp4 files (each file is a datapoint)...")
        try:
            hdf5_file_paths = extract_to_mp4_chunked(
                sequence_folder=Path(local_seq_dir),
                output_dir=out_dir,
                frame_rate=args.frame_rate,
                resolution=args.resolution,
            )
            # Get corresponding MP4 files
            mp4_file_paths = [Path(str(h5_path).replace('.h5', '.mp4')) for h5_path in hdf5_file_paths]
            print(f"✓ Processed sequence into {len(hdf5_file_paths)} hdf5 files and {len(mp4_file_paths)} mp4 files")
        except Exception as e:
            print(f"❌ Extraction failed for {key}: {e}")
            # Clean up downloaded sequence
            if os.path.isdir(local_seq_dir):
                shutil.rmtree(local_seq_dir)
            continue

        ########################################################
        ####  Step 3: Upload HDF5 and MP4 files to S3  ######
        ########################################################
        print(f"[{idx}/{len(keys)}] Step 3/4: Uploading HDF5 and MP4 files to S3...")
        # Upload HDF5 files
        for hdf5_path in hdf5_file_paths:
            s3_hdf5_path = f"{s3_prefix.rstrip('/')}/{Path(hdf5_path).name}"
            try:
                run_s3_upload_file(str(hdf5_path), s3_hdf5_path, aws_profile, aws_region)
                print(f"✓ Uploaded HDF5 to {s3_hdf5_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ S3 upload failed for HDF5 {hdf5_path}: {e}")
                continue

        # Upload MP4 files
        for mp4_path in mp4_file_paths:
            # Check if MP4 file exists before uploading
            if not os.path.exists(mp4_path):
                print(f"⚠️  Skipping upload - MP4 file does not exist: {mp4_path}")
                continue

            s3_mp4_path = f"{s3_prefix.rstrip('/')}/{Path(mp4_path).name}"
            try:
                run_s3_upload_file(str(mp4_path), s3_mp4_path, aws_profile, aws_region)
                print(f"✓ Uploaded MP4 to {s3_mp4_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ S3 upload failed for MP4 {mp4_path}: {e}")
                continue

        ########################################################
        ####  Step 4: Clean up local files  ######
        ########################################################
        print(f"[{idx}/{len(keys)}] Step 4/4: Cleaning up local files...")
        try:
            if os.path.isdir(local_seq_dir):
                shutil.rmtree(local_seq_dir)
                print(f"✓ Deleted sequence directory: {local_seq_dir}")
            # Clean up HDF5 files
            for hdf5_path in hdf5_file_paths:
                if os.path.exists(hdf5_path):
                    os.remove(hdf5_path)
                    print(f"✓ Deleted HDF5 file: {hdf5_path}")
            # Clean up MP4 files
            for mp4_path in mp4_file_paths:
                if os.path.exists(mp4_path):
                    os.remove(mp4_path)
                    print(f"✓ Deleted MP4 file: {mp4_path}")
        except Exception as e:
            print(f"⚠️  Cleanup warning for {key}: {e}")
        # also delete data_summary.json and download_summary.json if they exist
        data_summary_path = os.path.join(out_dir, 'data_summary.json')
        download_summary_path = os.path.join(out_dir, 'download_summary.json')
        if os.path.exists(data_summary_path):
            os.remove(data_summary_path)
            print(f"✓ Deleted data_summary.json: {data_summary_path} in {out_dir}")
        if os.path.exists(download_summary_path):
            os.remove(download_summary_path)
            print(f"✓ Deleted download_summary.json: {download_summary_path} in {out_dir}")

        print(f"✅ [{idx}/{len(keys)}] Successfully processed: {key}")

    print(f"\n{'='*80}")
    print(f"Processing complete! Processed {len(keys)} sequences.")
    print(f"{'='*80}")


if __name__ == "__main__":
    tyro.cli(main)


