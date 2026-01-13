import json
import os
import sys
import subprocess
import csv
from pathlib import Path
from typing import List, Optional, Set, Dict
from dataclasses import dataclass, field
import tyro
from extract_antego_data import extract_to_mp4_chunked_multiprocess
import shutil
import re
import threading
from queue import Queue
from multiprocessing import cpu_count
import filelock
import logging
from datetime import datetime


WORKSPACE_ROOT = "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset"
DEFAULT_URL_JSON = f"{WORKSPACE_ROOT}/Nymeria_download_urls.json"
DEFAULT_OUT_DIR = f"{WORKSPACE_ROOT}/temp-upload-folder"
DEFAULT_S3_PREFIX = "s3://far-research-internal/antzhan/nymeria/mp4"
DEFAULT_AWS_PROFILE = "far-compute"
DEFAULT_AWS_REGION = "us-west-2"

# Sentinel values to signal end of work for each stage
DOWNLOAD_COMPLETE = None
PROCESSING_COMPLETE = None

# CSV file for tracking sequences to skip (bad data or failed processing)
SEQUENCES_TO_SKIP_CSV = f"{WORKSPACE_ROOT}/sequences_to_skip.csv"


def setup_logging() -> logging.Logger:
    """Set up logging to both file and console with timestamps.

    Returns:
        Logger instance configured for the pipeline.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(WORKSPACE_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    # Configure logging
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    logger.handlers.clear()

    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler (info and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(threadName)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def load_sequences_to_skip(logger: Optional[logging.Logger] = None) -> Set[str]:
    """Load sequence keys to skip from the CSV file in SEQUENCES_TO_SKIP_CSV

    Returns:
        Set of sequence keys that should be skipped.
    """
    sequences = set()
    csv_path = SEQUENCES_TO_SKIP_CSV

    if not os.path.exists(csv_path):
        return sequences

    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'sequence_key' in row and row['sequence_key']:
                    sequences.add(row['sequence_key'])
    except Exception as e:
        msg = f"Warning: Could not load sequences_to_skip.csv: {e}"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    return sequences


def add_sequence_to_skip(sequence_key: str, reason: str, logger: Optional[logging.Logger] = None) -> None:
    """Add a failed sequence to the CSV file. Uses file locking to ensure thread-safe writes.

    Args:
        sequence_key: The sequence key to add (e.g., "20230607_s0_james_johnson_act0_e72nhq")
        reason: The reason for skipping (e.g., "download failed: HTTP 404")
        logger: Optional logger instance
    """
    csv_path = SEQUENCES_TO_SKIP_CSV
    lock_path = f"{csv_path}.lock"

    # Use file locking for thread safety
    lock = filelock.FileLock(lock_path, timeout=30)

    try:
        with lock:
            # Check if sequence already exists in CSV
            existing_sequences = load_sequences_to_skip(logger)
            if sequence_key in existing_sequences:
                msg = f"[SkipList] Sequence {sequence_key} already in skip list, not adding again"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                return

            # Check if file exists and has header
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['sequence_key', 'reason'])
                writer.writerow([sequence_key, reason])

            msg = f"[SkipList] Added {sequence_key} to skip list: {reason}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
    except filelock.Timeout:
        msg = f"[SkipList] Warning: Could not acquire lock to add {sequence_key} to skip list"
        if logger:
            logger.warning(msg)
        else:
            print(msg)
    except Exception as e:
        msg = f"[SkipList] Error adding {sequence_key} to skip list: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)

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


def load_all_sequence_keys(url_json_path: str, only_with_narrations: bool = True, logger: Optional[logging.Logger] = None) -> List[str]:
    """Load all sequence keys from the JSON file, optionally filtering for those with narrations.

    This function also removes sequences with bad data using the sequences_to_skip.csv file.
    """
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
        msg = f"Filtered to {len(keys)} sequences with language annotations (out of {len(sequences)} total)"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    else:
        # If not filtering for sequences with language annotations, return all sequences
        # NOTE: There are 1100 total sequences in the dataset, but only 864 of them have language annotations.
        keys = sorted(sequences.keys()) # keys is a length 1100 list of sequence names: ['20230607_s0_james_johnson_act0_e72nhq', '20230607_s0_james_johnson_act1_7xwm28', ...]

    # remove keys with bad data (loaded from CSV file)
    sequences_to_skip = load_sequences_to_skip(logger)
    original_count = len(keys)
    keys = [k for k in keys if k not in sequences_to_skip]
    skipped_count = original_count - len(keys)
    if skipped_count > 0:
        msg = f"Skipped {skipped_count} sequences from skip list (sequences_to_skip.csv)"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    return keys


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


def get_existing_sequence_keys_from_s3(s3_prefix: str, aws_profile: str, aws_region: str, min_files: int = 100) -> set:
    """List all files in S3 bucket and extract fully-processed sequence keys.

    Files in S3 are named like: {sequence_key}_{chunk_number}.h5 or {sequence_key}_{chunk_number}.mp4
    For example: 20230607_s0_james_johnson_act0_e72nhq_00000.h5

    A sequence is considered fully-processed if it has at least `min_files` h5 files
    AND at least `min_files` corresponding mp4 files.

    Args:
        s3_prefix: S3 path prefix (e.g., "s3://bucket/path")
        aws_profile: AWS profile name
        aws_region: AWS region
        min_files: Minimum number of h5 AND mp4 files required to consider a sequence complete (default: 100)

    Returns:
        Set of sequence keys that are fully processed.
    """
    from collections import defaultdict

    cmd = ["aws", "s3", "ls", s3_prefix.rstrip('/') + '/', "--region", aws_region] # ex: AWS_PROFILE=far-compute aws s3 ls s3://far-research-internal/antzhan/nymeria/mp4/ --region us-west-2
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        lines = result.stdout.strip().split('\n')
    except subprocess.CalledProcessError:
        # If the bucket/prefix doesn't exist or is empty, return empty set
        return set()

    # Track chunk numbers for h5 and mp4 files per sequence key
    h5_chunks = defaultdict(set)   # sequence_key -> set of chunk numbers (e.g., {"00000", "00001", ...})
    mp4_chunks = defaultdict(set)  # sequence_key -> set of chunk numbers

    for line in lines:
        if not line.strip():
            continue
        # S3 ls output format: "2024-01-01 12:00:00    12345 filename.ext"
        parts = line.split()
        if len(parts) >= 4:
            filename = parts[-1]  # Get the filename (last part)

            # Extract sequence key and chunk number
            # Pattern: {sequence_key}_{chunk_number}.{ext}
            if filename.endswith('.h5'):
                base_name = filename[:-3]  # Remove .h5
                match = re.match(r'^(.+)_(\d+)$', base_name)
                if match:
                    sequence_key, chunk_num = match.groups()
                    h5_chunks[sequence_key].add(chunk_num)
            elif filename.endswith('.mp4'):
                base_name = filename[:-4]  # Remove .mp4
                match = re.match(r'^(.+)_(\d+)$', base_name)
                if match:
                    sequence_key, chunk_num = match.groups()
                    mp4_chunks[sequence_key].add(chunk_num)

    # Only return sequences that have at least min_files MATCHING pairs (same chunk number for both h5 and mp4)
    fully_processed = set()
    all_keys = set(h5_chunks.keys()) | set(mp4_chunks.keys())
    for seq_key in all_keys:
        # Find chunk numbers that exist in BOTH h5 and mp4
        matching_chunks = h5_chunks[seq_key] & mp4_chunks[seq_key]
        if len(matching_chunks) >= min_files:
            fully_processed.add(seq_key)

    return fully_processed


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
    keys: List[str],        # Ex: ['20230607_s0_james_johnson_act0_e72nhq', '20230607_s0_james_johnson_act1_7xwm28', ...] (already filtered, excludes S3 existing)
    url_json_path: str,     # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/Nymeria_download_urls.json"
    out_dir: str,           # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    download_queue: Queue,  # Queue of DownloadedSequence objects
    total_keys: int,        # Ex: 100 (number of keys to process)
    logger: logging.Logger, # Logger instance
) -> None:
    """Stage 1: Downloads sequences in 'keys' and puts them in the download_queue as a DownloadedSequence object.

    This runs in a separate thread so that downloads can happen concurrently
    with processing. The queue has maxsize to limit disk usage.

    Note: 'keys' is already filtered to exclude sequences that exist in S3.
    """
    for idx, key in enumerate(keys, start=1): # Ex: '20230607_s0_james_johnson_act0_e72nhq'
        local_seq_dir = os.path.join(out_dir, key) # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq"
        logger.info(f"\n[Downloader] [{idx}/{total_keys}] Downloading: {key}")

        try:
            run_download(url_json_path, out_dir, key) # Downloads the nymeria sequence to the out_dir

            if os.path.isdir(local_seq_dir):
                # Put downloaded sequence in queue (blocks if queue is full)
                download_queue.put(DownloadedSequence(key=key, local_seq_dir=local_seq_dir, idx=idx))
                logger.info(f"[Downloader] [{idx}/{total_keys}] Queued for processing: {key}")
            else:
                logger.error(f"[Downloader] [{idx}/{total_keys}] Directory not found after download: {local_seq_dir}")
                add_sequence_to_skip(key, f"download failed: directory not created", logger)
                try:
                    contents = os.listdir(out_dir)
                    logger.debug(f"[Downloader] Current contents of {out_dir}: {contents[:20]}")
                except Exception:
                    pass
        except subprocess.CalledProcessError as e:
            logger.error(f"[Downloader] [{idx}/{total_keys}] Download failed for {key}: {e}")
            add_sequence_to_skip(key, f"download failed: {e}", logger)

    # Signal that all downloads are complete, the thread will exit after these lines are executed.
    download_queue.put(DOWNLOAD_COMPLETE)
    logger.info("[Downloader] All downloads complete, exiting thread")


def processor_thread(
    download_queue: Queue,
    upload_queue: Queue,
    args: 'DownloadUploadArgs',
    total_keys: int,
    logger: logging.Logger,
) -> None:
    """Stage 2: Processes downloaded sequences into HDF5/MP4 files and queues them for upload.

    This version uses multiprocessing to parallelize the processing of atomic actions
    within each sequence, significantly speeding up the processing stage.

    Takes DownloadedSequence objects from download_queue, processes them into HDF5/MP4 files
    using multiple worker processes, and puts ProcessedSequence objects into upload_queue.
    """
    logger.info("[Processor] PROCESSOR THREAD STARTED!")
    logger.info(f"[Processor] Using {args.num_processing_workers} worker processes for multiprocessing")
    out_dir = os.path.abspath(args.out_dir) # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    processed_count = 0

    while True:
        item: Optional[DownloadedSequence] = download_queue.get() # Queue.get() will block until an item is available in the queue.

        if item is DOWNLOAD_COMPLETE:
            logger.info("[Processor] Received download completion signal")
            # Signal to uploader that no more sequences are coming
            upload_queue.put(PROCESSING_COMPLETE)
            break

        key = item.key                      # Ex: "20230607_s0_james_johnson_act0_e72nhq"
        local_seq_dir = item.local_seq_dir  # Ex: "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq"
        idx = item.idx                      # Ex: 1

        logger.info(f"\n{'='*80}")
        logger.info(f"[Processor] [{idx}/{total_keys}] Processing: {key} (with {args.num_processing_workers} workers)")
        logger.info(f"{'='*80}")

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
            logger.info(f"[Processor] Processed into {len(hdf5_file_paths)} hdf5 + {len(mp4_file_paths)} mp4 files")

            # Queue for upload (blocks if upload_queue is full)
            upload_queue.put(ProcessedSequence(
                key=key,
                local_seq_dir=local_seq_dir,
                hdf5_file_paths=hdf5_file_paths,
                mp4_file_paths=mp4_file_paths,
                idx=idx,
            ))
            logger.info(f"[Processor] [{idx}/{total_keys}] Queued for upload: {key}")
            processed_count += 1

        except Exception as e:
            logger.error(f"[Processor] Extraction failed for {key}: {e}", exc_info=True)
            add_sequence_to_skip(key, f"processing failed: {e}", logger)
            # Clean up downloaded sequence on failure
            if os.path.isdir(local_seq_dir):
                logger.info(f"[Processor] Cleaning up sequence directory after failure: {local_seq_dir}")
                try:
                    shutil.rmtree(local_seq_dir)
                    logger.info(f"[Processor] Successfully deleted: {local_seq_dir}")
                except Exception as cleanup_error:
                    logger.error(f"[Processor] Failed to cleanup {local_seq_dir}: {cleanup_error}")
            continue

    logger.info(f"[Processor] Finished processing {processed_count} sequences, exiting thread")


def uploader_thread(
    upload_queue: Queue,
    args: 'DownloadUploadArgs',
    total_keys: int,
    logger: logging.Logger,
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
            logger.info("[Uploader] Received processing completion signal, exiting")
            break

        key = item.key
        local_seq_dir = item.local_seq_dir
        hdf5_file_paths = item.hdf5_file_paths
        mp4_file_paths = item.mp4_file_paths
        idx = item.idx

        logger.info(f"[Uploader] [{idx}/{total_keys}] Uploading: {key}")

        # Upload HDF5 files
        for hdf5_path in hdf5_file_paths:
            s3_hdf5_path = f"{s3_prefix.rstrip('/')}/{Path(hdf5_path).name}" # Ex: "s3://far-research-internal/antzhan/nymeria/mp4/20230607_s0_james_johnson_act0_e72nhq_00000.h5"
            # only upload if the hdf5 file has a corresponding mp4 file
            if not os.path.exists(str(hdf5_path).replace('.h5', '.mp4')):
                logger.warning(f"[Uploader] Skipping - HDF5 does not have a corresponding MP4: {hdf5_path}")
                with open(os.path.join(out_dir, 'orphaned_hdf5_files.txt'), 'a') as f:
                    f.write(f"{hdf5_path}\n")
                continue
            try:
                run_s3_upload_file(str(hdf5_path), s3_hdf5_path, aws_profile, aws_region)
                logger.debug(f"[Uploader] Uploaded HDF5 to {s3_hdf5_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"[Uploader] S3 upload failed for HDF5 {hdf5_path}: {e}")

        # Upload MP4 files
        for mp4_path in mp4_file_paths:
            # Check if MP4 file exists before uploading
            if not os.path.exists(mp4_path):
                logger.warning(f"[Uploader] Skipping - MP4 does not exist: {mp4_path}")
                continue

            s3_mp4_path = f"{s3_prefix.rstrip('/')}/{Path(mp4_path).name}"
            try:
                run_s3_upload_file(str(mp4_path), s3_mp4_path, aws_profile, aws_region)
                logger.debug(f"[Uploader] Uploaded MP4 to {s3_mp4_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"[Uploader] S3 upload failed for MP4 {mp4_path}: {e}")

        # Clean up local files
        logger.info(f"[Uploader] [{idx}/{total_keys}] Cleaning up: {key}")
        cleanup_errors = []
        try:
            # Delete sequence directory (raw downloaded data)
            if os.path.isdir(local_seq_dir):
                shutil.rmtree(local_seq_dir)
                logger.info(f"[Uploader] Deleted sequence directory: {local_seq_dir}")
            else:
                logger.warning(f"[Uploader] Sequence directory does not exist (already deleted?): {local_seq_dir}")

            # Delete HDF5 files
            for hdf5_path in hdf5_file_paths:
                if os.path.exists(hdf5_path):
                    try:
                        os.remove(hdf5_path)
                        logger.debug(f"[Uploader] Deleted HDF5 file: {hdf5_path}")
                    except Exception as e:
                        cleanup_errors.append(f"HDF5 {hdf5_path}: {e}")
                        logger.error(f"[Uploader] Failed to delete HDF5 {hdf5_path}: {e}")

            # Delete MP4 files
            for mp4_path in mp4_file_paths:
                if os.path.exists(mp4_path):
                    try:
                        os.remove(mp4_path)
                        logger.debug(f"[Uploader] Deleted MP4 file: {mp4_path}")
                    except Exception as e:
                        cleanup_errors.append(f"MP4 {mp4_path}: {e}")
                        logger.error(f"[Uploader] Failed to delete MP4 {mp4_path}: {e}")
        except Exception as e:
            logger.error(f"[Uploader] Cleanup error for {key}: {e}", exc_info=True)
            cleanup_errors.append(f"General cleanup: {e}")

        # also delete data_summary.json and download_summary.json if they exist
        data_summary_path = os.path.join(out_dir, 'data_summary.json')
        download_summary_path = os.path.join(out_dir, 'download_summary.json')
        try:
            if os.path.exists(data_summary_path):
                os.remove(data_summary_path)
                logger.debug(f"[Uploader] Deleted data_summary.json: {data_summary_path}")
            if os.path.exists(download_summary_path):
                os.remove(download_summary_path)
                logger.debug(f"[Uploader] Deleted download_summary.json: {download_summary_path}")
        except Exception as e:
            logger.error(f"[Uploader] Failed to delete summary files: {e}")

        uploaded_count += 1
        if cleanup_errors:
            logger.warning(f"[Uploader] [{idx}/{total_keys}] Completed with {len(cleanup_errors)} cleanup errors: {key}")
        else:
            logger.info(f"[Uploader] [{idx}/{total_keys}] Completed successfully: {key}")

    logger.info(f"[Uploader] Finished uploading {uploaded_count} sequences")


@dataclass
class DownloadUploadArgs:
    url_json: str = DEFAULT_URL_JSON    # "/home/ubuntu/sky_workdir/nymeria_dataset/Nymeria_download_urls.json"
    """Path to `Nymeria_download_urls.json` file"""

    out_dir: str = DEFAULT_OUT_DIR      # "/home/ubuntu/sky_workdir/nymeria_dataset/temp-upload-folder"
    """Local output directory for temporary downloads"""

    limit: int = -1
    """Number of NEW sequences to process (after filtering out those already in S3). If -1, process all remaining sequences."""

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
    download_queue_size: int = 2
    """Max number of sequences to download ahead. Actual disk usage can be queue_size + 2 (includes current download and current processing)."""

    upload_queue_size: int = 5
    """Max number of processed sequences waiting to be uploaded. Each sequence contains 100-200+ h5/mp4 file pairs.
    WARNING: Setting this too high can leave many orphaned files if the script is terminated abruptly."""

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
    # Set up logging first
    logger = setup_logging()

    url_json_path = os.path.abspath(args.url_json)
    out_dir = os.path.abspath(args.out_dir)
    s3_prefix = args.s3_prefix
    aws_profile = args.aws_profile
    aws_region = args.aws_region
    limit = args.limit
    only_with_narrations = args.only_with_narrations

    os.makedirs(out_dir, exist_ok=True)

    # Load all sequence keys from JSON that have good data
    all_keys = load_all_sequence_keys(url_json_path, only_with_narrations, logger)
    if not all_keys:
        logger.error("No sequences found in the provided JSON.")
        sys.exit(1)
    logger.info(f"Found {len(all_keys)} total sequences in JSON that have good data.")

    # Get existing sequence keys from S3 to skip already-uploaded sequences
    logger.info("Checking S3 for existing sequences...")
    existing_keys = get_existing_sequence_keys_from_s3(s3_prefix, aws_profile, aws_region)
    logger.info(f"Found {len(existing_keys)} existing sequences in S3.")

    # Filter out sequences already in S3, then apply limit
    keys_to_process = [k for k in all_keys if k not in existing_keys] # keys_to_process is a list of all new sequences to process.
    logger.info(f"Remaining sequences to process: {len(keys_to_process)}")
    if limit != -1:
        keys = keys_to_process[:limit]
        logger.info(f"Applying limit={limit}: will process {len(keys)} sequences")
    else:
        keys = keys_to_process
    if not keys:
        logger.info("No new sequences to process (all already in S3).")
        sys.exit(0)

    logger.info(f"\nWill process {len(keys)} sequences.")
    logger.info(f"Temp directory: {out_dir}")
    logger.info(f"S3 destination: {s3_prefix}")
    logger.info(f"Extraction params: all datapoints @ {args.frame_rate}fps")
    logger.info(f"Pipeline queues: download={args.download_queue_size} sequences, upload={args.upload_queue_size} sequences")
    logger.info(f"Processing workers: {args.num_processing_workers} (multiprocessing)\n")

    # Create queues for the 3-stage pipeline
    download_queue: Queue[Optional[DownloadedSequence]] = Queue(maxsize=args.download_queue_size)
    upload_queue: Queue[Optional[ProcessedSequence]] = Queue(maxsize=args.upload_queue_size)

    # Create threads for each stage
    downloader = threading.Thread(
        target=downloader_thread,
        args=(keys, url_json_path, out_dir, download_queue, len(keys), logger),
        daemon=True,
        name="DownloaderThread"
    )

    processor = threading.Thread(
        target=processor_thread,
        args=(download_queue, upload_queue, args, len(keys), logger),
        daemon=True,
        name="ProcessorThread"
    )

    uploader = threading.Thread(
        target=uploader_thread,
        args=(upload_queue, args, len(keys), logger),
        daemon=True,
        name="UploaderThread"
    )

    logger.info("Starting 3-stage pipeline with MULTIPROCESSING...")
    logger.info("   [Stage 1 - Downloader] downloads raw sequence data")
    logger.info(f"   [Stage 2 - Processor]  extracts to hdf5 + mp4 ({args.num_processing_workers} parallel workers)")
    logger.info("   [Stage 3 - Uploader]   uploads to S3 and cleans up\n")

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
        logger.warning("Warning: Downloader thread did not exit cleanly")
    if processor.is_alive():
        logger.warning("Warning: Processor thread did not exit cleanly")

    logger.info(f"\n{'='*80}")
    logger.info(f"Pipeline complete!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    tyro.cli(main)
