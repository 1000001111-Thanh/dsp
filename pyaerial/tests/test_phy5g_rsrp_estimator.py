# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test RsrpEstimator."""
# Ensure that all the test vectors are available in TEST_VECTOR_DIR.
import glob
import pytest
from pytest import TEST_VECTOR_DIR

import h5py as h5
import numpy as np

from aerial.phy5g.algorithms import RsrpEstimator


test_case_numbers = [7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213,
                     7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7227,
                     7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7242,
                     7243, 7244, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256,
                     7257, 7258, 7259, 7260, 7261]

tc_to_skip = [7239, 7247]

test_case_numbers = set(list(test_case_numbers)) - set(tc_to_skip)


@pytest.mark.parametrize(
    "test_case_number",
    test_case_numbers,
    ids=[int(test_case_number) for test_case_number in test_case_numbers]
)
def test_rsrp_estimator(pusch_config, cuda_stream, test_case_number):
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    enable_pusch_tdi = input_file["gnb_pars"]["TdiMode"][0]

    pusch_configs = pusch_config(input_file)

    ch_est = []
    ree_diag_inv = []
    for ue_grp_idx in range(len(input_file["ueGrp_pars"])):
        ch_est_ = np.array(input_file[f"reference_H_est{ue_grp_idx}"])["re"] + 1j * \
            np.array(input_file[f"reference_H_est{ue_grp_idx}"])["im"]
        if ch_est_.ndim == 3:
            ch_est_ = ch_est_[..., None].transpose(2, 1, 0, 3)
        else:
            ch_est_ = ch_est_.transpose(3, 2, 1, 0)
        ch_est.append(ch_est_)

        ree_diag = np.array(input_file[f"reference_Ree{ue_grp_idx}"])
        try:
            ree_diag_inv_ = 1. / ree_diag
        except TypeError:
            ree_diag_inv_ = 1. / ree_diag["re"]
        if ree_diag_inv_.ndim == 2:
            ree_diag_inv_ = ree_diag_inv_[None, ...]
        ree_diag_inv.append(ree_diag_inv_)

    noise_var_pre_eq = np.array(input_file["reference_noiseVardBPerUe"])[0]

    rsrp_estimator = RsrpEstimator(
        num_rx_ant=num_rx_ant,
        enable_pusch_tdi=enable_pusch_tdi,
        cuda_stream=cuda_stream
    )

    rsrp, pre_eq_sinr, post_eq_sinr = rsrp_estimator.estimate(
        channel_est=ch_est,
        ree_diag_inv=ree_diag_inv,
        noise_var_pre_eq=noise_var_pre_eq,
        pusch_configs=pusch_configs
    )

    ref_rsrp = np.array(input_file["reference_rsrpdB"])[0]
    assert np.allclose(rsrp, ref_rsrp, rtol=0.1)

    ref_post_eq_sinr = np.array(input_file["reference_postEqSinrdB"])[0]
    assert np.allclose(post_eq_sinr, ref_post_eq_sinr, rtol=0.1)
