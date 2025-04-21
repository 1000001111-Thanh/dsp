# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test the whole LPDC coding chain end to end."""
import pytest

import numpy as np

from aerial.phy5g.ldpc import LdpcEncoder
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import LdpcRateMatch
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import get_mcs
from aerial.phy5g.ldpc import random_tb
from aerial.phy5g.ldpc import code_block_segment
from aerial.phy5g.ldpc import get_crc_len


@pytest.mark.parametrize("enable_scrambling, mcs, num_prb, rv", [

    # Note that RV1 and RV2 need a lower code rate to be self-decodable.
    (True, 3, 100, 0),
    (True, 1, 100, 1),
    (True, 1, 100, 2),
    (True, 3, 100, 3),
    (False, 3, 100, 0),
    (False, 1, 100, 1),
    (False, 1, 100, 2),
    (False, 3, 100, 3),
    (True, 10, 6, 0),
    (True, 1, 6, 1),
    (True, 1, 6, 2),
    (True, 10, 6, 3),
    (False, 10, 6, 0),
    (False, 1, 6, 1),
    (False, 1, 6, 2),
    (False, 10, 6, 3),
    (True, 10, 100, 0),
    (True, 1, 100, 1),
    (True, 1, 100, 2),
    (True, 10, 100, 3),
    (False, 10, 100, 0),
    (False, 1, 100, 1),
    (False, 1, 100, 2),
    (False, 10, 100, 3),
    (True, 10, 272, 0),
    (True, 1, 272, 1),
    (True, 1, 272, 2),
    (True, 10, 272, 3),
    (False, 10, 272, 0),
    (False, 1, 272, 1),
    (False, 1, 272, 2),
    (False, 10, 272, 3),
])
def test_ldpc_endtoend(cuda_stream, enable_scrambling, mcs, num_prb, rv):

    start_sym = 0
    num_ofdm_symbols = 14
    num_layers = 1

    rnti = 20000               # UE RNTI
    data_scid = 41             # Data scrambling ID
    cinit = (rnti << 15) + data_scid
    dmrs_sym = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    ldpc_encoder = LdpcEncoder(cuda_stream=cuda_stream)
    ldpc_decoder = LdpcDecoder(cuda_stream=cuda_stream)
    ldpc_rate_match = LdpcRateMatch(enable_scrambling=enable_scrambling, cuda_stream=cuda_stream)
    ldpc_derate_match = LdpcDeRateMatch(
        enable_scrambling=enable_scrambling,
        cuda_stream=cuda_stream
    )

    mod_order, code_rate = get_mcs(mcs)

    # Generate a random transport block (in bits).
    transport_block = random_tb(
        mod_order=mod_order,
        code_rate=code_rate,
        dmrs_syms=dmrs_sym,
        num_prbs=num_prb,
        start_sym=start_sym,
        num_symbols=num_ofdm_symbols,
        num_layers=num_layers,
        return_bits=True
    )
    tb_size = transport_block.shape[0]

    crc_length = get_crc_len(tb_size)
    crc = np.random.randint(0, 1, size=crc_length, dtype=np.uint8)
    transport_block = np.concatenate((transport_block, crc))

    code_rate /= 1024.
    code_blocks = code_block_segment(tb_size, transport_block, code_rate)

    coded_bits = ldpc_encoder.encode(
        input_data=code_blocks,
        tb_size=tb_size,
        code_rate=code_rate,
        redundancy_version=rv
    )

    num_data_sym = (num_ofdm_symbols - np.array(dmrs_sym).sum())
    rate_match_len = num_data_sym * num_prb * 12 * num_layers * mod_order
    rate_matched_bits = ldpc_rate_match.rate_match(
        input_data=coded_bits,
        tb_size=tb_size,
        code_rate=code_rate,
        rate_match_len=rate_match_len,
        mod_order=mod_order,
        num_layers=num_layers,
        redundancy_version=rv,
        cinit=cinit
    )

    rx_bits = 1 - 2. * rate_matched_bits

    derate_matched_bits = ldpc_derate_match.derate_match(
        input_llrs=[rx_bits],
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        rate_match_lengths=[rate_match_len],
        mod_orders=[mod_order],
        num_layers=[num_layers],
        redundancy_versions=[rv],
        ndis=[1],
        cinits=[cinit]
    )

    decoded_bits = ldpc_decoder.decode(
        input_llrs=derate_matched_bits,
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        redundancy_versions=[rv],
        rate_match_lengths=[rate_match_len]
    )[0].astype(np.uint8)

    assert np.array_equal(decoded_bits, code_blocks)
