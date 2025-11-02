import json
import os
import sys
import subprocess
from pathlib import Path
from typing import List
from dataclasses import dataclass
import tyro


WORKSPACE_ROOT = "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset"
DEFAULT_URL_JSON = f"{WORKSPACE_ROOT}/Nymeria_download_urls.json"
DEFAULT_OUT_DIR = f"{WORKSPACE_ROOT}/temp-upload-folder"
DEFAULT_S3_PREFIX = "s3://far-research-internal/antzhan/nymeria/test100"
DEFAULT_AWS_PROFILE = "far-compute"
DEFAULT_AWS_REGION = "us-west-2"


def load_first_n_sequence_keys(url_json_path: str, limit: int) -> List[str]:
    with open(url_json_path, "r") as f:
        data = json.load(f)
    sequences = data.get("sequences", {})
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



@dataclass
class DownloadUploadArgs:

    url_json: str = DEFAULT_URL_JSON # "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset"
    """Path to `Nymeria_download_urls.json` file"""

    out_dir: str = DEFAULT_OUT_DIR # "/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/temp-upload-folder"
    """Local output directory"""

    limit: int = 100 # 100
    """Number of sequences to process (alphabetically)"""

    s3_prefix: str = DEFAULT_S3_PREFIX # "s3://far-research-internal/antzhan/nymeria/test100"
    """Destination S3 prefix"""

    aws_profile: str = DEFAULT_AWS_PROFILE
    """AWS profile name to use"""

    aws_region: str = DEFAULT_AWS_REGION
    """AWS region for S3 operations"""


def main(args: DownloadUploadArgs) -> None:
    url_json_path = os.path.abspath(args.url_json)
    out_dir = os.path.abspath(args.out_dir)
    s3_prefix = args.s3_prefix
    aws_profile = args.aws_profile
    aws_region = args.aws_region
    limit = args.limit

    os.makedirs(out_dir, exist_ok=True)

    keys = load_first_n_sequence_keys(url_json_path, limit)
    if not keys:
        print("No sequences found in the provided JSON.")
        sys.exit(1)

    print(f"Found {len(keys)} sequences to process. Output -> {out_dir}, S3 -> {s3_prefix}")

    for idx, key in enumerate(keys, start=1):
        local_seq_dir = os.path.join(out_dir, key) # '/home/ANT.AMAZON.COM/antzhan/lab42/src/FAR-nymeria-dataset/temp-upload-folder/20230607_s0_james_johnson_act0_e72nhq'
        print(f"[{idx}/{len(keys)}] Downloading: {key}")
        try:
            run_download(url_json_path, out_dir, key)
        except subprocess.CalledProcessError as e:
            print(f"Download failed for {key}: {e}")
            continue

        if not os.path.isdir(local_seq_dir):
            # If the expected directory isn't present, list for debugging and skip upload
            print(f"Expected directory not found: {local_seq_dir}")
            try:
                contents = os.listdir(out_dir)
                print(f"Current contents of {out_dir}: {contents[:20]}{' ...' if len(contents) > 20 else ''}")
            except Exception:
                pass
            continue

        print(f"[{idx}/{len(keys)}] Uploading to S3: {key}")
        try:
            run_s3_upload(local_seq_dir, s3_prefix, aws_profile, aws_region)
        except subprocess.CalledProcessError as e:
            print(f"Upload failed for {key}: {e}")
            continue

        print(f"[{idx}/{len(keys)}] Done: {key}")


if __name__ == "__main__":
    tyro.cli(main)


