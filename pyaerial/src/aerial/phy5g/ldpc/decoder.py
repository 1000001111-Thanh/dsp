# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - LDPC decoding."""
from typing import List
from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.ldpc.util import get_code_block_size
from aerial.phy5g.params import PuschConfig


class LdpcDecoder:
    """LDPC decoder.

    This class supports decoding of LDPC code blocks encoded following TS 38.212. It uses
    cuPHY accelerated LDPC decoding routines under the hood.
    """

    def __init__(self,
                 num_iterations: int = 10,
                 throughput_mode: bool = False,
                 cuda_stream: int = None) -> None:
        """Initialize LdpcDecoder.

        Args:
            num_iterations (int): Number of LDPC decoder iterations. Default: 10.
            throughput_mode (bool): Enable throughput mode. Default: False.
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        self.num_iterations = num_iterations
        self.throughput_mode = throughput_mode

        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        # Create cuPHY LDPC decoder object.
        self.pycuphy_ldpc_decoder = pycuphy.LdpcDecoder(  # pylint: disable=no-member
            self.cuda_stream
        )

        self.pycuphy_ldpc_decoder.set_num_iterations(num_iterations)
        self.pycuphy_ldpc_decoder.set_throughput_mode(throughput_mode)

    def decode(self,
               *,
               input_llrs: List[np.ndarray],
               pusch_configs: List[PuschConfig] = None,
               tb_sizes: List[int] = None,
               code_rates: List[float] = None,
               redundancy_versions: List[int] = None,
               rate_match_lengths: List[int] = None,
               num_iterations: int = None) -> List[np.ndarray]:
        """Decode function for LDPC decoder.

        The decoder outputs decoded code blocks which can be further concatenated into
        the received transport block using :class:`~aerial.phy5g.ldpc.crc_check.CrcChecker`.

        Args:
            input_llrs (List[np.ndarray]): Input LLRs per UE, each array is a N x C array of 32-bit
                floats, N being the number of LLRs per code block and C being the number of code
                blocks.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given.
            tb_sizes (List[int]): Transport block size in bits, without CRC, per UE.
            code_rates (List[float]): Target code rates per UE.
            redundancy_versions (List[int]): Redundancy version, i.e. 0, 1, 2, or 3, per UE.
            rate_match_lengths (int): Number of rate matching output bits of each UE.
                This is equal to N.
            num_iterations (int): Number of LDPC iterations. If not given, use the default from
                the constructor.

        Returns:
            List[np.ndarray]: The decoded bits in a numpy array.
        """
        # If PuschConfigs given, read the other parameters from that (the rest are ignored).
        if pusch_configs is not None:
            tb_sizes = []
            code_rates = []
            redundancy_versions = []
            rate_match_lengths = []
            for pusch_config in pusch_configs:
                tb_sizes += [ue_config.tb_size * 8 for ue_config in pusch_config.ue_configs]
                code_rates += [ue_config.code_rate / 10240.
                               for ue_config in pusch_config.ue_configs]
                redundancy_versions += [ue_config.rv for ue_config in pusch_config.ue_configs]
                num_data_sym = (np.array(pusch_config.dmrs_syms[
                    pusch_config.start_sym:pusch_config.start_sym + pusch_config.num_symbols
                ]) == 0).sum()
                num_ues = len(pusch_config.ue_configs)
                rate_match_lengths += [
                    num_data_sym * pusch_config.ue_configs[ue].mod_order * pusch_config.num_prbs *
                    12 * pusch_config.ue_configs[ue].layers for ue in range(num_ues)
                ]
        # Make sure all parameters are given in this case.
        else:
            if tb_sizes is None:
                raise ValueError("Argument tb_sizes is not set!")
            if code_rates is None:
                raise ValueError("Argument code_rates is not set!")
            if redundancy_versions is None:
                raise ValueError("Argument redundancy_versions is not set!")
            if rate_match_lengths is None:
                raise ValueError("Argument rate_match_lengths is not set!")

        # Reset the LDPC num of iterations if necessary
        if (num_iterations is not None) and (num_iterations != self.num_iterations):
            self.pycuphy_ldpc_decoder.set_num_iterations(num_iterations)
            self.num_iterations = num_iterations

        # C++ wrapper function.
        decoded_bits_ = self.pycuphy_ldpc_decoder.decode(
            [llr.astype(np.float32) for llr in input_llrs],
            [np.uint32(tb_size) for tb_size in tb_sizes],
            [np.float32(code_rate) for code_rate in code_rates],
            [np.uint32(rv) for rv in redundancy_versions],
            [np.uint32(rate_match_len) for rate_match_len in rate_match_lengths]
        )

        # Remove the possible padding to maximum code block size.
        decoded_bits = []
        for ue_idx, bits in enumerate(decoded_bits_):
            cb_size = get_code_block_size(tb_sizes[ue_idx], code_rates[ue_idx])
            decoded_bits.append(bits[:cb_size, :])

        return decoded_bits

    def set_num_iterations(self, num_iterations: int) -> None:
        """Set a particular value for the number of iterations to be run.

        Args:
            num_iterations (int): Value of the number of iterations.
        """
        self.num_iterations = num_iterations
        self.pycuphy_ldpc_decoder.set_num_iterations(num_iterations)

    def set_throughput_mode(self, throughput_mode: bool) -> None:
        """Enable throughput mode.

        Args:
            throughput_mode (bool): Enable (True) throughput mode.
        """
        self.throughput_mode = throughput_mode
        self.pycuphy_ldpc_decoder.set_throughput_mode(throughput_mode)

    def get_soft_bits(self) -> List[np.ndarray]:
        """Get the soft bit output from the decoder.

        Returns:
            List[np.ndarray]: The soft bits in a numpy array.
        """
        soft_bits = self.pycuphy_ldpc_decoder.get_soft_outputs()
        return soft_bits
