# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""CUDA-related utilities."""
from typing import Any

from cuda import cudart  # type: ignore


def get_cuda_stream() -> cudart.cudaStream_t:
    """Return a CUDA stream.

    Returns:
        cudart.cudaStream_t: A new CUDA stream.
    """
    cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
    return cuda_stream


def check_cuda_errors(result: cudart.cudaError_t) -> Any:
    """Check CUDA errors.

    Args:
        result (cudart.cudaError_t): CUDA error value.
    """
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]
