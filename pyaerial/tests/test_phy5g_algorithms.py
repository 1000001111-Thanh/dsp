# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test the PUSCH Rx pipeline built from separate components."""
import glob
import pytest
from pytest import TEST_VECTOR_DIR

import h5py as h5
import numpy as np

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.algorithms import ChannelEqualizer
from aerial.phy5g.algorithms import NoiseIntfEstimator
from aerial.phy5g.algorithms import RsrpEstimator
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import CrcChecker


test_case_numbers = [7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213,
                     7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7225, 7227, 7229,
                     7230, 7231, 7232, 7233, 7236, 7242, 7249, 7251, 7252, 7253, 7254, 7255, 7256,
                     7258, 7259, 7261]


# pylint: disable=too-many-locals
@pytest.mark.parametrize(
    "test_case_number",
    test_case_numbers,
    ids=[int(test_case_number) for test_case_number in test_case_numbers]
)
def test_pusch_rx_components(pusch_config,
                             cuda_stream,
                             test_case_number):
    """Test PUSCH Rx components."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    # Extract the test vector parameters.
    rx_slot = np.array(input_file["DataRx"])["re"] + 1j * np.array(input_file["DataRx"])["im"]
    rx_slot = rx_slot.transpose(2, 1, 0)

    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    enable_pusch_tdi = input_file["gnb_pars"]["TdiMode"][0]
    eq_coeff_algo = input_file["gnb_pars"]["eqCoeffAlgoIdx"][0]
    slot = np.array(input_file["gnb_pars"]["slotNumber"])[0]

    tb_pars = np.array(input_file["tb_pars"])

    pusch_configs = pusch_config(input_file)

    num_ues = input_file["ueGrp_pars"]["nUes"].sum()

    # Build the components
    channel_estimator = ChannelEstimator(
        num_rx_ant=num_rx_ant,
        ch_est_algo=1,
        cuda_stream=cuda_stream
    )

    noise_intf_estimator = NoiseIntfEstimator(
        num_rx_ant=num_rx_ant,
        eq_coeff_algo=eq_coeff_algo,
        cuda_stream=cuda_stream
    )

    channel_equalizer = ChannelEqualizer(
        num_rx_ant=num_rx_ant,
        eq_coeff_algo=eq_coeff_algo,
        enable_pusch_tdi=enable_pusch_tdi,
        cuda_stream=cuda_stream
    )

    rsrp_estimator = RsrpEstimator(
        num_rx_ant=num_rx_ant,
        enable_pusch_tdi=enable_pusch_tdi,
        cuda_stream=cuda_stream
    )

    derate_match = LdpcDeRateMatch(enable_scrambling=True, cuda_stream=cuda_stream)
    decoder = LdpcDecoder(cuda_stream=cuda_stream)
    crc_checker = CrcChecker(cuda_stream=cuda_stream)

    # Run the Rx data through the receiver components.
    ch_est = channel_estimator.estimate(
        rx_slot=rx_slot,
        slot=slot,
        pusch_configs=pusch_configs
    )

    lw_inv, noise_var_pre_eq = noise_intf_estimator.estimate(
        rx_slot=rx_slot,
        channel_est=ch_est,
        slot=slot,
        pusch_configs=pusch_configs
    )

    llrs, _ = channel_equalizer.equalize(
        rx_slot=rx_slot,
        channel_est=ch_est,
        lw_inv=lw_inv,
        noise_var_pre_eq=noise_var_pre_eq,
        pusch_configs=pusch_configs
    )

    ree_diag_inv = channel_equalizer.ree_diag_inv
    rsrp, _, post_eq_sinr = rsrp_estimator.estimate(
        channel_est=ch_est,
        ree_diag_inv=ree_diag_inv,
        noise_var_pre_eq=noise_var_pre_eq,
        pusch_configs=pusch_configs
    )

    # Check RSRP and post-eq SINR estimates.
    ref_rsrp = np.array(input_file["reference_rsrpdB"])[0]
    assert np.allclose(rsrp, ref_rsrp, atol=0.5)

    ref_post_eq_sinr = np.array(input_file["reference_postEqSinrdB"])[0]
    assert np.allclose(post_eq_sinr, ref_post_eq_sinr, rtol=0.1)

    # Run the rest of the pipeline.
    coded_blocks = derate_match.derate_match(
        input_llrs=llrs,
        pusch_configs=pusch_configs
    )

    code_blocks = decoder.decode(
        input_llrs=coded_blocks,
        pusch_configs=pusch_configs
    )

    tb, tb_crcs = crc_checker.check_crc(
        input_bits=code_blocks,
        pusch_configs=pusch_configs
    )

    ref_tb_payloads = np.array(input_file["tb_payload"])
    offset = 0
    for ue in range(num_ues):
        assert tb_crcs[ue][0] == 0, f"UE#{ue} CRC check failed!"

        tbs = tb_pars["nTbByte"][ue]
        assert np.array_equal(tb[ue], ref_tb_payloads[offset:offset + tbs, 0]), \
            f"Wrong payload for UE#{ue}!"
        offset += tbs

        # 4-byte alignment in the reference payload.
        offset = (offset + 3) // 4 * 4
