import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import tyro
from extract_antego_data import extract_to_mp4_chunked_multiprocess
import shutil
import re
import threading
from queue import Queue
from multiprocessing import cpu_count


WORKSPACE_ROOT = "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset"
DEFAULT_URL_JSON = f"{WORKSPACE_ROOT}/Nymeria_download_urls.json"
DEFAULT_OUT_DIR = f"{WORKSPACE_ROOT}/temp-upload-folder"
DEFAULT_S3_PREFIX = "s3://far-research-internal/antzhan/nymeria/mp4"
DEFAULT_AWS_PROFILE = "far-compute"
DEFAULT_AWS_REGION = "us-west-2"

# Sentinel values to signal end of work for each stage
DOWNLOAD_COMPLETE = None
PROCESSING_COMPLETE = None


@dataclass
class DownloadedSequence:
    """Represents a successfully downloaded sequence ready for processing"""
    key: str            # Ex: "20230607_s0_james_johnson_act0_e72nhq"
    local_seq_dir: str  # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq"
    idx: int            # Ex: 1


@dataclass
class ProcessedSequence:
    """Represents a processed sequence ready for upload to S3 (atomic actions ahve been extracted to HDF5/MP4 files for this sequence)"""
    key: str                        # Ex: "20230607_s0_james_johnson_act0_e72nhq"
    local_seq_dir: str              # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq"
    hdf5_file_paths: List[Path]     # List of hdf5 file paths to upload
    mp4_file_paths: List[Path]      # List of mp4 file paths to upload
    idx: int                        # Ex: 1


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

    return keys if limit == -1 else keys[:limit]


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
    """Upload a directory to S3"""
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

    # Log the command to a text file
    cmd_str = f"AWS_PROFILE={aws_profile} {' '.join(cmd)}"
    log_file = os.path.join(WORKSPACE_ROOT, "batch_mp4_s3_upload_directory_commands.txt")
    with open(log_file, "a") as f:
        f.write(cmd_str + "\n")

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

    # Log the command to a text file
    cmd_str = f"AWS_PROFILE={aws_profile} {' '.join(cmd)}"
    log_file = os.path.join(WORKSPACE_ROOT, "batch_mp4_s3_upload_file_commands.txt")
    with open(log_file, "a") as f:
        f.write(cmd_str + "\n")

    subprocess.run(cmd, check=True, env=env)


def downloader_thread(
    keys: List[str],        # Ex: ['20230607_s0_james_johnson_act0_e72nhq', '20230607_s0_james_johnson_act1_7xwm28', ...]
    existing_keys: set,     # set of sequence keys already in S3
    url_json_path: str,     # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/Nymeria_download_urls.json"
    out_dir: str,           # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    download_queue: Queue,  # Queue of DownloadedSequence objects
    total_keys: int,        # Ex: 1100
) -> None:
    """Stage 1: Downloads sequences in 'keys' and puts them in the download_queue as a DownloadedSequence object.

    This runs in a separate thread so that downloads can happen concurrently
    with processing. The queue has maxsize to limit disk usage.
    """
    for idx, key in enumerate(keys, start=1): # Ex: '20230607_s0_james_johnson_act0_e72nhq'
        # Skip if already in S3
        if key in existing_keys:
            print(f"[Downloader] [{idx}/{total_keys}] Skipping (already in S3): {key}")
            continue

        local_seq_dir = os.path.join(out_dir, key) # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq"
        print(f"\n[Downloader] [{idx}/{total_keys}] Downloading: {key}")

        try:
            run_download(url_json_path, out_dir, key) # Downloads the nymeria sequence to the out_dir

            if os.path.isdir(local_seq_dir):
                # Put downloaded sequence in queue (blocks if queue is full)
                download_queue.put(DownloadedSequence(key=key, local_seq_dir=local_seq_dir, idx=idx))
                print(f"[Downloader] [{idx}/{total_keys}] Queued for processing: {key}")
            else:
                print(f"[Downloader] [{idx}/{total_keys}] Directory not found after download: {local_seq_dir}")
                try:
                    contents = os.listdir(out_dir)
                    print(f"[Downloader] Current contents of {out_dir}: {contents[:20]}")
                except Exception:
                    pass
        except subprocess.CalledProcessError as e:
            print(f"[Downloader] [{idx}/{total_keys}] Download failed for {key}: {e}")

    # Signal that all downloads are complete, the thread will exit after these lines are executed.
    download_queue.put(DOWNLOAD_COMPLETE)
    print("[Downloader] All downloads complete, exiting thread")


def processor_thread(
    download_queue: Queue,
    upload_queue: Queue,
    args: 'DownloadUploadArgs',
    total_keys: int,
) -> None:
    """Stage 2: Processes downloaded sequences into HDF5/MP4 files and queues them for upload.

    This version uses multiprocessing to parallelize the processing of atomic actions
    within each sequence, significantly speeding up the processing stage.

    Takes DownloadedSequence objects from download_queue, processes them into HDF5/MP4 files
    using multiple worker processes, and puts ProcessedSequence objects into upload_queue.
    """
    print("[Processor] PROCESSOR THREAD STARTED!", flush=True)
    print(f"[Processor] Using {args.num_processing_workers} worker processes for multiprocessing", flush=True)
    out_dir = os.path.abspath(args.out_dir) # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    processed_count = 0

    while True:
        item: Optional[DownloadedSequence] = download_queue.get() # Queue.get() will block until an item is available in the queue.

        if item is DOWNLOAD_COMPLETE:
            print("[Processor] Received download completion signal")
            # Signal to uploader that no more sequences are coming
            upload_queue.put(PROCESSING_COMPLETE)
            break

        key = item.key                      # Ex: "20230607_s0_james_johnson_act0_e72nhq"
        local_seq_dir = item.local_seq_dir  # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq"
        idx = item.idx                      # Ex: 1

        print(f"\n{'='*80}")
        print(f"[Processor] [{idx}/{total_keys}] Processing: {key} (with {args.num_processing_workers} workers)")
        print(f"{'='*80}")

        try:
            # Use the multiprocessing version for faster processing
            hdf5_file_paths = extract_to_mp4_chunked_multiprocess(
                sequence_folder=Path(local_seq_dir),# Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq"
                output_dir=out_dir,                 # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
                frame_rate=args.frame_rate,         # Ex: 30.0
                resolution=args.resolution,         # Ex: 1408
                num_workers=args.num_processing_workers,  # Number of parallel workers for atomic action processing
            )
            # Get corresponding MP4 files
            mp4_file_paths = [Path(str(h5_path).replace('.h5', '.mp4')) for h5_path in hdf5_file_paths]
            print(f"[Processor] Processed into {len(hdf5_file_paths)} hdf5 + {len(mp4_file_paths)} mp4 files")

            # Queue for upload (blocks if upload_queue is full)
            upload_queue.put(ProcessedSequence(
                key=key,
                local_seq_dir=local_seq_dir,
                hdf5_file_paths=hdf5_file_paths,
                mp4_file_paths=mp4_file_paths,
                idx=idx,
            ))
            print(f"[Processor] [{idx}/{total_keys}] Queued for upload: {key}", flush=True)
            processed_count += 1

        except Exception as e:
            print(f"[Processor] Extraction failed for {key}: {e}")
            # Clean up downloaded sequence on failure
            if os.path.isdir(local_seq_dir):
                shutil.rmtree(local_seq_dir)
            continue

    print(f"[Processor] Finished processing {processed_count} sequences, exiting thread")


def uploader_thread(
    upload_queue: Queue,
    args: 'DownloadUploadArgs',
    total_keys: int,
) -> None:
    """Stage 3: Uploads processed files to S3 and cleans up local files.

    Takes ProcessedSequence objects from upload_queue, uploads the HDF5/MP4 files to S3,
    and then cleans up all local files (sequence directory + processed files).
    """
    out_dir = os.path.abspath(args.out_dir) # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    s3_prefix = args.s3_prefix              # Ex: "s3://far-research-internal/antzhan/nymeria/mp4"
    aws_profile = args.aws_profile          # Ex: "far-compute"
    aws_region = args.aws_region            # Ex: "us-west-2"

    uploaded_count = 0

    while True:
        item: Optional[ProcessedSequence] = upload_queue.get() # Queue.get() will block until an item is available in the queue.

        if item is PROCESSING_COMPLETE:
            print("[Uploader] Received processing completion signal, exiting")
            break

        key = item.key
        local_seq_dir = item.local_seq_dir
        hdf5_file_paths = item.hdf5_file_paths
        mp4_file_paths = item.mp4_file_paths
        idx = item.idx

        print(f"[Uploader] [{idx}/{total_keys}] Uploading: {key}")

        # Upload HDF5 files
        for hdf5_path in hdf5_file_paths:
            s3_hdf5_path = f"{s3_prefix.rstrip('/')}/{Path(hdf5_path).name}" # Ex: "s3://far-research-internal/antzhan/nymeria/mp4/20230607_s0_james_johnson_act0_e72nhq_00000.h5"
            # only upload if the hdf5 file has a corresponding mp4 file
            if not os.path.exists(str(hdf5_path).replace('.h5', '.mp4')):
                print(f"[Uploader] Skipping - HDF5 does not have a corresponding MP4: {hdf5_path}")
                with open(os.path.join(out_dir, 'orphaned_hdf5_files.txt'), 'a') as f:
                    f.write(f"{hdf5_path}\n")
                continue
            try:
                run_s3_upload_file(str(hdf5_path), s3_hdf5_path, aws_profile, aws_region)
                print(f"[Uploader] Uploaded HDF5 to {s3_hdf5_path}")
            except subprocess.CalledProcessError as e:
                print(f"[Uploader] S3 upload failed for HDF5 {hdf5_path}: {e}")

        # Upload MP4 files
        for mp4_path in mp4_file_paths:
            # Check if MP4 file exists before uploading
            if not os.path.exists(mp4_path):
                print(f"[Uploader] Skipping - MP4 does not exist: {mp4_path}")
                continue

            s3_mp4_path = f"{s3_prefix.rstrip('/')}/{Path(mp4_path).name}"
            try:
                run_s3_upload_file(str(mp4_path), s3_mp4_path, aws_profile, aws_region)
                print(f"[Uploader] Uploaded MP4 to {s3_mp4_path}")
            except subprocess.CalledProcessError as e:
                print(f"[Uploader] S3 upload failed for MP4 {mp4_path}: {e}")

        # Clean up local files
        print(f"[Uploader] [{idx}/{total_keys}] Cleaning up: {key}")
        try:
            # Delete sequence directory (raw downloaded data)
            if os.path.isdir(local_seq_dir):
                shutil.rmtree(local_seq_dir)
                print(f"[Uploader] Deleted sequence directory: {local_seq_dir}")
            # Delete HDF5 files
            for hdf5_path in hdf5_file_paths:
                if os.path.exists(hdf5_path):
                    os.remove(hdf5_path)
                    print(f"[Uploader] Deleted HDF5 file: {hdf5_path}")
            # Delete MP4 files
            for mp4_path in mp4_file_paths:
                if os.path.exists(mp4_path):
                    os.remove(mp4_path)
                    print(f"[Uploader] Deleted MP4 file: {mp4_path}")
        except Exception as e:
            print(f"[Uploader] Cleanup warning for {key}: {e}")

        # also delete data_summary.json and download_summary.json if they exist
        data_summary_path = os.path.join(out_dir, 'data_summary.json')
        download_summary_path = os.path.join(out_dir, 'download_summary.json')
        if os.path.exists(data_summary_path):
            os.remove(data_summary_path)
            print(f"[Uploader] Deleted data_summary.json: {data_summary_path}")
        if os.path.exists(download_summary_path):
            os.remove(download_summary_path)
            print(f"[Uploader] Deleted download_summary.json: {download_summary_path}")

        uploaded_count += 1
        print(f"[Uploader] [{idx}/{total_keys}] Completed: {key}", flush=True)

    print(f"[Uploader] Finished uploading {uploaded_count} sequences")


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

    s3_prefix: str = DEFAULT_S3_PREFIX  # "s3://far-research-internal/antzhan/nymeria/mp4"
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

    # Multithreading parameters
    download_queue_size: int = 3
    """Max number of sequences to download ahead. Actual disk usage can be queue_size + 2 (includes current download and current processing)."""

    upload_queue_size: int = 1000
    """Max sequences waiting to be uploaded."""

    # Multiprocessing parameters for the processing stage
    num_processing_workers: int = 4
    """Number of parallel worker processes for processing atomic actions within each sequence.
    Each worker creates its own NymeriaDataProvider instance. Set to a higher number for faster
    processing, but be mindful of memory usage (each worker loads the sequence data).
    Recommended: 4-8 for machines with 32GB+ RAM, 2-4 for machines with 16GB RAM."""


def main(args: DownloadUploadArgs) -> None:
    """Downloads, processes, and uploads Nymeria sequences using a 3-stage pipeline.

    Architecture (3 concurrent stages with multiprocessing in the processor):
    ┌─────────────┐   download_queue   ┌─────────────────────────────────────┐   upload_queue   ┌─────────────┐
    │  Downloader │ ─────────────────► │           Processor Thread          │ ───────────────► │  Uploader   │
    │   Thread    │                    │  ┌────────────────────────────────┐ │                  │   Thread    │
    └─────────────┘                    │  │   Multiprocessing Pool         │ │                  └─────────────┘
         │                             │  │  ┌────────┐ ┌────────┐ ┌────┐  │ │                       │
         │ downloads raw               │  │  │Worker 1│ │Worker 2│ │... │  │ │                       │ uploads to S3
         │ sequence data               │  │  └────────┘ └────────┘ └────┘  │ │                       │ and cleans up
                                       │  │   (each processes atomic       │ │
                                       │  │    actions in parallel)        │ │
                                       │  └────────────────────────────────┘ │
                                       └─────────────────────────────────────┘
                                              extracts to hdf5 + mp4 (FAST!)

    This allows all 3 stages to run concurrently:
    - While uploading sequence N-1, we can process sequence N and download sequence N+1
    - Processing is parallelized using multiprocessing for significant speedup
    """
    url_json_path = os.path.abspath(args.url_json)
    out_dir = os.path.abspath(args.out_dir)
    s3_prefix = args.s3_prefix
    aws_profile = args.aws_profile
    aws_region = args.aws_region
    limit = args.limit
    only_with_narrations = args.only_with_narrations

    os.makedirs(out_dir, exist_ok=True)
    keys = load_first_n_sequence_keys(url_json_path, limit, only_with_narrations)
    if not keys:
        print("No sequences found in the provided JSON.")
        sys.exit(1)

    print(f"Found {len(keys)} sequences to process.")
    print(f"Temp directory: {out_dir}")
    print(f"S3 destination: {s3_prefix}")
    print(f"Extraction params: all datapoints @ {args.frame_rate}fps")
    print(f"Pipeline queues: download={args.download_queue_size}, upload={args.upload_queue_size}")
    print(f"Processing workers: {args.num_processing_workers} (multiprocessing)\n")

    # Get existing sequence keys from S3 to skip already-uploaded sequences
    print("Checking S3 for existing sequences...")
    existing_keys = get_existing_sequence_keys_from_s3(s3_prefix, aws_profile, aws_region)
    print(f"Found {len(existing_keys)} existing sequences in S3.\n")

    # Create queues for the 3-stage pipeline
    download_queue: Queue[Optional[DownloadedSequence]] = Queue(maxsize=args.download_queue_size)
    upload_queue: Queue[Optional[ProcessedSequence]] = Queue(maxsize=args.upload_queue_size)

    # Create threads for each stage
    downloader = threading.Thread(
        target=downloader_thread,
        args=(keys, existing_keys, url_json_path, out_dir, download_queue, len(keys)),
        daemon=True,
        name="DownloaderThread"
    )

    processor = threading.Thread(
        target=processor_thread,
        args=(download_queue, upload_queue, args, len(keys)),
        daemon=True,
        name="ProcessorThread"
    )

    uploader = threading.Thread(
        target=uploader_thread,
        args=(upload_queue, args, len(keys)),
        daemon=True,
        name="UploaderThread"
    )

    print("Starting 3-stage pipeline with MULTIPROCESSING...")
    print("   [Stage 1 - Downloader] downloads raw sequence data")
    print(f"   [Stage 2 - Processor]  extracts to hdf5 + mp4 ({args.num_processing_workers} parallel workers)")
    print("   [Stage 3 - Uploader]   uploads to S3 and cleans up\n")

    # Start all threads
    downloader.start()
    processor.start()
    uploader.start()

    # Wait for uploader to finish (it's the last stage)
    uploader.join()

    # Clean up other threads (should already be done)
    downloader.join(timeout=5.0)
    processor.join(timeout=5.0)

    if downloader.is_alive():
        print("Warning: Downloader thread did not exit cleanly")
    if processor.is_alive():
        print("Warning: Processor thread did not exit cleanly")

    print(f"\n{'='*80}")
    print(f"Pipeline complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    tyro.cli(main)
