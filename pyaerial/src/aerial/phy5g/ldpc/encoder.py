# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - LDPC encoding."""
from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors


class LdpcEncoder:
    """LDPC encoder.

    This class provides encoding of transmitted transport block bits using LDPC coding
    following TS 38.212. The encoding process is GPU accelerated using cuPHY routines.
    As the input, the transport blocks are assumed to be attached with the CRC and
    segmented to code blocks (as per TS 38.212).
    """

    def __init__(self,
                 num_profiling_iterations: int = 0,
                 puncturing: bool = True,
                 max_num_code_blocks: int = 152,
                 cuda_stream: int = None) -> None:
        """Initialize LdpcEncoder.

        Initialization does all the necessary memory allocations for cuPHY.

        Args:
            num_profiling_iterations (int): Number of profiling iterations.
                Set to 0 to disable profiling. Default: 0.
            puncturing (bool): Whether to puncture the systematic bits (2Zc). Default: True.
            max_num_code_blocks (int): Maximum number of code blocks. Memory is allocated based
                on this. Default: 152.
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        self.num_profiling_iterations = num_profiling_iterations
        self.puncturing = puncturing

        # Memory allocation. Allocate maximum possible for a single transport block.
        data_size = 4
        max_lifting_size = 384
        max_input_size = 22 * max_lifting_size * max_num_code_blocks * data_size
        max_output_size = 68 * max_lifting_size * max_num_code_blocks * data_size

        self.input_device_ptr = check_cuda_errors(cudart.cudaMalloc(max_input_size))
        self.temp_input_host_ptr = check_cuda_errors(cudart.cudaMallocHost(max_input_size))
        self.output_device_ptr = check_cuda_errors(cudart.cudaMalloc(max_output_size))
        self.output_host_ptr = check_cuda_errors(cudart.cudaMallocHost(max_output_size))

        # Create LDPC encoder object.
        self.pycuphy_ldpc_encoder = pycuphy.LdpcEncoder(  # pylint: disable=no-member
            self.input_device_ptr,
            self.temp_input_host_ptr,
            self.output_device_ptr,
            self.output_host_ptr,
            self.cuda_stream
        )

        self.pycuphy_ldpc_encoder.set_profiling_iterations(num_profiling_iterations)
        self.pycuphy_ldpc_encoder.set_puncturing(puncturing)

    def encode(self,
               input_data: np.ndarray,
               tb_size: int,
               code_rate: float,
               redundancy_version: int) -> np.ndarray:
        """Encode function for LDPC encoder.

        The input to this function is code blocks, meaning that the code block segmentation
        is expected to be done before calling this function. Code block segmentation can be done
        using :func:`~aerial.phy5g.ldpc.util.code_block_segment`.

        Args:
            input_data (np.ndarray): The input code blocks as a K x C array where K is the
                number of input bits per code block (including CRCs) and C is the number
                of code blocks. The dtype of the input array must be `np.float32`.
            tb_size (int): Transport block size in bits, without CRC.
            code_rate (float): Target code rate.
            redundancy_version (int): Redundancy version, 0, 1, 2, or 3.

        Returns:
            np.ndarray: Encoded bits as a N x C array where N is the number of
            encoded bits per code block.
        """
        # Profiling start.
        if self.num_profiling_iterations:
            start = check_cuda_errors(cudart.cudaEventCreate())
            stop = check_cuda_errors(cudart.cudaEventCreate())

            check_cuda_errors(cudart.cudaEventRecord(start, self.cuda_stream))

        # C++ wrapper function.
        num_input_bits, num_code_blocks = input_data.shape
        coded_bits = self.pycuphy_ldpc_encoder.encode(
            np.ascontiguousarray(input_data).astype(np.float32),
            np.uint32(tb_size),
            np.float32(code_rate),
            int(redundancy_version)
        )

        # Profiling stop.
        if self.num_profiling_iterations:
            check_cuda_errors(cudart.cudaEventRecord(stop, self.cuda_stream))
            check_cuda_errors(cudart.cudaEventSynchronize(stop))

            time = check_cuda_errors(cudart.cudaEventElapsedTime(start, stop))
            print(f"Total time from Python {time * 1000} us.")

            ext_tput = (
                (num_input_bits * num_code_blocks * self.num_profiling_iterations) / time
            ) / 1000000
            print(
                f"External throughput is {ext_tput} Gbps for {self.num_profiling_iterations} runs."
            )

        return coded_bits

    def set_profiling_iterations(self, num_profiling_iterations: int) -> None:
        """Set a particular value for the number of profiling iterations to be run.

        Args:
            num_profiling_iterations (int): Value of the number of profiling iterations.
        """
        self.num_profiling_iterations = num_profiling_iterations
        self.pycuphy_ldpc_encoder.set_profiling_iterations(num_profiling_iterations)

    def set_puncturing(self, puncturing: bool) -> None:
        """Set puncturing flag.

        Args:
            puncturing (bool): Whether to puncture the systematic bits (2*Zc). Default: True.
        """
        self.puncturing = puncturing
        self.pycuphy_ldpc_encoder.set_puncturing(puncturing)

    def __del__(self) -> None:
        """Destroy function."""
        # Free allocated memory.
        check_cuda_errors(cudart.cudaFreeHost(self.temp_input_host_ptr))
        check_cuda_errors(cudart.cudaFree(self.input_device_ptr))
        check_cuda_errors(cudart.cudaFree(self.output_device_ptr))
        check_cuda_errors(cudart.cudaFreeHost(self.output_host_ptr))
