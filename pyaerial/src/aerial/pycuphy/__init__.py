# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial - The cuPHY backend."""
import ctypes
import os

# Load shared libraries.
libcuphy_path = os.path.dirname(os.path.realpath(__file__))

dynamic_libs = ["libfmtlog-shared.so", "libnvlog.so", "libcuphy.so", "libchanModels.so"]
for lib in dynamic_libs:
    so = os.path.join(libcuphy_path, lib)
    if os.path.isfile(so):
        ctypes.cdll.LoadLibrary(so)

# Disable lints due to import being in the wrong place.
from ._pycuphy import *  # type: ignore  # noqa: F403  # pylint: disable=C0413
