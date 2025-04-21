# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test LDPC encoding-decoding chain on random data."""
import numpy as np

from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import LdpcEncoder
from aerial.phy5g.ldpc import add_crc_len


def test_lpdc_random_data(cuda_stream):

    # Random data generation
    # Data based on TV x031 - Lifting Size Zc = 384
    tb_size = 3824
    tb_size_with_crc = add_crc_len(tb_size)  # pylint: disable=invalid-name

    # Provide appropriate code rate for Base Graph selection
    code_rate = 666 / 1024

    # Randomly generated data.
    input_uncoded_data = np.random.randint(2, size=(tb_size_with_crc, 1))

    ldpc_encoder = LdpcEncoder(puncturing=False, cuda_stream=cuda_stream)
    # Run LDPC encoding.
    coded_output = ldpc_encoder.encode(
        input_data=input_uncoded_data,
        tb_size=tb_size,
        code_rate=code_rate,
        redundancy_version=0
    )

    # BPSK modulation and addition of AWGN.
    sigma = 10.
    bpsk_mod_data = 1 / (np.sqrt(2)) * (1 - 2 * coded_output)
    llr = 2 * bpsk_mod_data / (sigma**2)

    ldpc_decoder = LdpcDecoder(cuda_stream=cuda_stream)
    decoded_output = ldpc_decoder.decode(
        input_llrs=[llr],
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        rate_match_lengths=[len(llr)],
        redundancy_versions=[0]
    )[0]

    assert np.array_equal(input_uncoded_data, decoded_output)
