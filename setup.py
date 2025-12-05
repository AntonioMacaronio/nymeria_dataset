# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# This setup.py has been modified to install the nymeria_dataloader module.
# It does not do anything else but simply allows utilization of the nymeria_dataloader module in the nymeria package.
#
# NOTE: torch is installed via conda in environment.yml (not here) because pymomentum
# requires CUDA-enabled PyTorch from conda-forge, which must match the libtorch version.

# If you are using the nymeria_dataloader module, you need to install the nymeria package in whatever environment you are using for which you need the nymeria_dataloader module.
# For example, If I want to finetune a model on the Nymeria dataset, I would need to install the nymeria package in the environment I am using for finetuning, which should already have torch with CUDA installed.
from setuptools import find_packages, setup

setup(
    name="nymeria",
    version="0.0.1",
    packages=find_packages(),
    author="Lingni Ma",
    author_email="lingni.ma@meta.com",
    description="The official repo to support the Nymeria dataset",
    python_requires=">=3.10",
    install_requires=[
        "viser",
        "click",
        "requests",
        "tqdm",
        # Dependencies for nymeria_dataloader
        "opencv-python",
        "pandas",
        "h5py",
        "hdf5plugin",  # For LZ4-compressed HDF5 files
        "numpy",
        "decord", # installed on CPU-only because determining right codec version for GPU-enabled PyTorch is tricky. This may be fast enough
        # torch w/ CUDA should already be installed if you are using the nymeria package through setup.py
    ],
)
