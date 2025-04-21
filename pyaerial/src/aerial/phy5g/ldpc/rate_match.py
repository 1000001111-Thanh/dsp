# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - LDPC rate matching."""
import math
from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors


# Maximum number of rate matched bits per code block
MAX_NUM_RATE_MATCHED_BITS_PER_CB = 256000


class LdpcRateMatch:
    """LDPC rate matching."""

    def __init__(self,
                 enable_scrambling: bool = True,
                 num_profiling_iterations: int = 0,
                 max_num_code_blocks: int = 152,
                 cuda_stream: int = None) -> None:
        """Initialize LdpcRateMatch.

        Initialization does all the necessary memory allocations for cuPHY.

        Args:
            enable_scrambling (bool): Whether to enable scrambling after code block concatenation.
            num_profiling_iterations (int): Number of profiling iterations. Set to 0 to disable
                profiling. Default: 0 (no profiling).
            max_num_code_blocks (int): Maximum number of code blocks. Memory will be allocated
                based on this number.
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        # Memory allocation.
        max_num_input_bits = 66 * 384  # Per code block, 66 * Zc.
        max_num_input_bits *= max_num_code_blocks
        # The bits are packed to 32-bit integers.
        max_num_input_bits = int(math.ceil(max_num_input_bits / 32.) * 32)
        num_input_bytes = max_num_input_bits / 8

        max_num_output_bits = MAX_NUM_RATE_MATCHED_BITS_PER_CB * max_num_code_blocks
        max_num_output_bytes = int(math.ceil(max_num_output_bits / 32.) * 32) / 8

        self.input_device_ptr = check_cuda_errors(cudart.cudaMalloc(num_input_bytes))
        self.input_host_ptr = check_cuda_errors(cudart.cudaMallocHost(num_input_bytes))
        self.output_device_ptr = check_cuda_errors(cudart.cudaMalloc(max_num_output_bytes))
        self.temp_output_host_ptr = check_cuda_errors(
            cudart.cudaMallocHost(max_num_output_bytes)
        )

        # Final output is one float per bit (the output numpy array).
        # Full allocation, 256QAM, four layers, float
        max_rate_match_len = 273 * 12 * (14 - 1) * 8 * 4 * 4
        self.output_host_ptr = check_cuda_errors(
            cudart.cudaMallocHost(max_rate_match_len)
        )

        # Create pycuphy LDPC rate match object.
        self.pycuphy_ldpc_rate_match = pycuphy.LdpcRateMatch(  # pylint: disable=no-member
            self.input_device_ptr,
            self.output_device_ptr,
            self.input_host_ptr,
            self.temp_output_host_ptr,
            self.output_host_ptr,
            enable_scrambling,
            self.cuda_stream
        )

        self.num_profiling_iterations = num_profiling_iterations
        if num_profiling_iterations > 0:
            self.set_profiling_iterations(num_profiling_iterations)

    def rate_match(self,
                   input_data: np.ndarray,
                   tb_size: int,
                   code_rate: float,
                   rate_match_len: int,
                   mod_order: int,
                   num_layers: int,
                   redundancy_version: int,
                   cinit: int) -> np.ndarray:
        """LDPC rate matching function.

        This function does rate matching of LDPC code blocks following TS 38.212. If scrambling
        is enabled, it also scrambles the rate matched bits. In this case the `c_init` value
        needs to be set to an appropriate scrambling sequence initialization value.

        Args:
            input_data (np.ndarray): Input bits as a N x C numpy array with dtype `np.float32`,
                where N is the number of bits per code block and C is the number of code
                blocks.
            tb_size (int): Transport block size in bits without CRC.
            code_rate (float): Code rate.
            rate_match_len (int): Number of rate matching output bits.
            mod_order (int): Modulation order.
            num_layers (int): Number of layers.
            redundancy_version (int): Redundancy version, i.e. 0, 1, 2, or 3.
            cinit (int): The `c_init` value used for initializing scrambling.

        Returns:
            np.ndarray: Rate matched bits.
        """
        # Profiling start.
        if self.num_profiling_iterations:
            start = check_cuda_errors(cudart.cudaEventCreate())
            stop = check_cuda_errors(cudart.cudaEventCreate())
            check_cuda_errors(cudart.cudaEventRecord(start, self.cuda_stream))

        # C++ wrapper function.
        rate_matched_bits = self.pycuphy_ldpc_rate_match.rate_match(
            input_data.astype(np.float32),
            np.uint32(tb_size),
            np.float32(code_rate),
            np.uint32(rate_match_len),
            np.uint8(mod_order),
            np.uint8(num_layers),
            np.uint8(redundancy_version),
            np.uint32(cinit)
        )

        # Profiling stop.
        if self.num_profiling_iterations:
            check_cuda_errors(cudart.cudaEventRecord(stop, self.cuda_stream))
            check_cuda_errors(cudart.cudaEventSynchronize(stop))

            time = check_cuda_errors(cudart.cudaEventElapsedTime(start, stop))
            print(f"Total time from Python {time * 1000} us.")

            ext_tput = (rate_match_len * self.num_profiling_iterations) / time / 1000000
            print(
                f"External throughput is {ext_tput} Gbps for {self.num_profiling_iterations} runs."
            )

        return rate_matched_bits

    def set_profiling_iterations(self, num_profiling_iterations: int) -> None:
        """Set a particular value for the number of profiling iterations to be run.

        Args:
            num_profiling_iterations (int): Value of the number of profiling iterations.
        """
        self.num_profiling_iterations = num_profiling_iterations
        self.pycuphy_ldpc_rate_match.set_profiling_iterations(num_profiling_iterations)

    def __del__(self) -> None:
        """Destroy function."""
        # Free allocated memory.
        check_cuda_errors(cudart.cudaFreeHost(self.input_host_ptr))
        check_cuda_errors(cudart.cudaFreeHost(self.temp_output_host_ptr))
        check_cuda_errors(cudart.cudaFreeHost(self.output_host_ptr))
        check_cuda_errors(cudart.cudaFree(self.input_device_ptr))
        check_cuda_errors(cudart.cudaFree(self.output_device_ptr))
