# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Setup file for PyAerial package."""
import os
import sys

import platform
import setuptools

sys.path.insert(0, os.path.abspath("./src/aerial"))
import version_aerial  # pylint: disable=E0401,C0413

machine = platform.machine()
version = version_aerial.release

setuptools.setup(
    name="pyaerial",
    version=version,
    author="NVIDIA",
    description="NVIDIA pyAerial library",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    package_data={"": [
        "version_aerial.py",
        f"_pycuphy.cpython-310-{machine}-linux-gnu.so",
        "libcuphy.so",
        "libnvlog.so",
        "libfmtlog-shared.so",
        "libchanModels.so",
        "chest_coeffs/cuPhyChEstCoeffs.h5"
    ]},
    python_requires=">=3.7",
    zip_safe=False,
)
