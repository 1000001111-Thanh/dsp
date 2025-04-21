# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test ChannelEqualizer."""
# Ensure that all the test vectors are available in TEST_VECTOR_DIR.
import glob
import pytest
from pytest import TEST_VECTOR_DIR

import h5py as h5
import numpy as np

from aerial.phy5g.algorithms import ChannelEqualizer


test_case_numbers = [7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213,
                     7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7227,
                     7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7242,
                     7243, 7244, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256,
                     7257, 7258, 7259, 7260, 7261]

tc_to_skip = [7239, 7246, 7247]

test_case_numbers = set(list(test_case_numbers)) - set(tc_to_skip)


@pytest.mark.parametrize(
    "test_case_number",
    test_case_numbers,
    ids=[int(test_case_number) for test_case_number in test_case_numbers]
)
def test_channel_equalizer(pusch_config, cuda_stream, test_case_number):
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    rx_slot = np.array(input_file["DataRx"])["re"] + 1j * np.array(input_file["DataRx"])["im"]
    rx_slot = rx_slot.transpose(2, 1, 0)

    ch_est = []
    lw_inv = []
    for ue_grp_idx in range(len(input_file["ueGrp_pars"])):

        ch_est_ = np.array(input_file[f"reference_H_est{ue_grp_idx}"])["re"] + 1j * \
            np.array(input_file[f"reference_H_est{ue_grp_idx}"])["im"]
        if ch_est_.ndim == 3:
            ch_est_ = ch_est_[..., None].transpose(2, 1, 0, 3)
        else:
            ch_est_ = ch_est_.transpose(3, 2, 1, 0)
        ch_est.append(ch_est_)

        try:
            noise_cov = np.array(input_file[f"reference_nCov{ue_grp_idx}"]["re"]) + 1j * \
                np.array(input_file[f"reference_nCov{ue_grp_idx}"]["im"])
        except ValueError:
            noise_cov = np.array(input_file[f"reference_nCov{ue_grp_idx}"])

        if noise_cov.ndim == 4:
            noise_cov = np.mean(noise_cov, axis=0)
        lw_inv_ = np.linalg.inv(np.linalg.cholesky(noise_cov))
        if lw_inv_.ndim == 3:
            lw_inv_ = lw_inv_.transpose(1, 2, 0)
        lw_inv.append(lw_inv_)

    noise_var_pre_eq = np.array(input_file["reference_noiseVardBPerUe"])

    pusch_configs = pusch_config(input_file)

    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    enable_pusch_tdi = input_file["gnb_pars"]["TdiMode"][0]
    eq_coeff_algo = input_file["gnb_pars"]["eqCoeffAlgoIdx"][0]
    num_layers = input_file["tb_pars"]["numLayers"]
    mod_orders = input_file["tb_pars"]["qamModOrder"]

    channel_equalizer = ChannelEqualizer(
        num_rx_ant=num_rx_ant,
        eq_coeff_algo=eq_coeff_algo,
        enable_pusch_tdi=enable_pusch_tdi,
        cuda_stream=cuda_stream
    )

    llr, eq_sym = channel_equalizer.equalize(
        rx_slot=rx_slot,
        channel_est=ch_est,
        lw_inv=lw_inv,
        noise_var_pre_eq=noise_var_pre_eq,
        pusch_configs=pusch_configs
    )

    # Check all UE groups.
    for ue_idx in range(len(llr)):

        # Check equalized symbols.
        ref_eq_sym = np.array(input_file[f"reference_X_est{ue_idx}"])
        ref_eq_sym = ref_eq_sym["re"] + 1j * ref_eq_sym["im"]
        if ref_eq_sym.ndim == 2:
            ref_eq_sym = ref_eq_sym.transpose()[None]
        else:
            ref_eq_sym = ref_eq_sym.transpose(0, 2, 1)

        assert np.allclose(eq_sym[ue_idx], ref_eq_sym, rtol=5e-2)

        # Check LLRs. Just check the sign as the magnitude can be quite different.
        ref_llr = np.array(input_file[f"reference_eqOutLLRs{ue_idx}"])
        ref_llr = ref_llr.transpose(3, 2, 1, 0)

        assert np.all(np.sign(llr[ue_idx][:mod_orders[ue_idx], :num_layers[ue_idx], ...]) ==
                      np.sign(ref_llr[:mod_orders[ue_idx], ...]))
