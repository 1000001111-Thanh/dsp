# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - CRC checking."""
from typing import List
from typing import Tuple
from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.params import PuschConfig


class CrcChecker:
    """CRC checking.

    This class supports decoding the code block CRCs, desegmenting code blocks together,
    assembling the transport block and also finally decoding the transport block CRCs.
    It uses cuPHY accelerated CRC routines under the hood.
    """

    def __init__(self, cuda_stream: int = None) -> None:
        """Initialize CrcChecker.

        Args:
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        self.crc_checker = pycuphy.CrcChecker(  # pylint: disable=no-member
            self.cuda_stream
        )

        self.cb_crcs = None

    def check_crc(self,
                  *,
                  input_bits: List[np.ndarray],
                  pusch_configs: List[PuschConfig] = None,
                  tb_sizes: List[int] = None,
                  code_rates: List[float] = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """CRC checking.

        This method takes LDPC decoder output as its input, checks the code block CRCs,
        desegments code blocks, combines them into a transport block and checks the
        transport block CRC. It returns the transport block payloads without CRC, as well
        as the transport block CRC check results. The code block CRC results are stored
        as well and may be queried separately.

        Args:
            input_bits (List[np.ndarray]): LDPC decoder outputs per UE, each array is a K x C array
                of 32-bit floats, K being the number of bits per code block and C being the number
                of code blocks.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given.
            tb_sizes (List[int]): Transport block size in bits, without CRC, per UE.
            code_rates (List[float]): Target code rates per UE.

        Returns:
            List[np.ndarray], np.ndarray: A tuple containing:

            - *List[np.ndarray]*:
              Transport block payloads in bytes, without CRC, for each UE.

            - *np.ndarray*:
              Transport block CRC check results for each UE.
        """
        max_code_block_size = 8448

        if pusch_configs is not None:
            tb_sizes = []
            code_rates = []
            for pusch_config in pusch_configs:
                tb_sizes += [ue_config.tb_size * 8 for ue_config in pusch_config.ue_configs]
                code_rates += [ue_config.code_rate / 10240.
                               for ue_config in pusch_config.ue_configs]
        # Make sure all parameters are given in this case.
        else:
            if tb_sizes is None:
                raise ValueError("Argument tb_sizes is not set!")
            if code_rates is None:
                raise ValueError("Argument code_rates is not set!")

        # cuPHY wants the LDPC output / CRC input extended to maximum number of info bits K
        # and stacked together.
        tot_num_code_blocks = sum(bits.shape[1] for bits in input_bits)
        crc_input = np.zeros(
            (max_code_block_size, tot_num_code_blocks),
            dtype=np.float32
        )
        idx = 0
        for ue_bits in input_bits:
            crc_input[:ue_bits.shape[0], idx:idx + ue_bits.shape[1]] = ue_bits.astype(np.float32)
            idx += ue_bits.shape[1]

        tb_payloads = self.crc_checker.check_crc(
            crc_input,
            [np.uint32(tb_size) for tb_size in tb_sizes],
            [np.float32(code_rate) for code_rate in code_rates],
        )

        self.cb_crcs = self.crc_checker.get_cb_crcs()

        tb_crcs = self.crc_checker.get_tb_crcs()

        return tb_payloads, tb_crcs
