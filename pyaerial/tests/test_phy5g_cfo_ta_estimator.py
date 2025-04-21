# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test CfoTaEstimator."""
# Ensure that all the test vectors are available in a folder title 'GPU_test_input'.
# This folder should be present in the same working directory as aerial_python.
import glob
import pytest
from pytest import TEST_VECTOR_DIR

import h5py as h5
import numpy as np

from aerial.phy5g.algorithms import CfoTaEstimator

test_case_numbers = [7401, 7402, 7403, 7404, 7405, 7406, 7408, 7409]


@pytest.mark.parametrize(
    "test_case_number",
    test_case_numbers,
    ids=[int(test_case_number) for test_case_number in test_case_numbers]
)
def test_cfo_ta_estimator(pusch_config, cuda_stream, test_case_number):

    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PUSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    ch_est = np.array(input_file["reference_H_est0"])["re"] + \
        1j * np.array(input_file["reference_H_est0"])["im"]
    if ch_est.ndim == 3:
        ch_est = ch_est[..., None].transpose(2, 1, 0, 3)
    else:
        ch_est = ch_est.transpose(3, 2, 1, 0)

    num_rx_ant = input_file["gnb_pars"]["nRx"][0]
    mu = input_file["gnb_pars"]["mu"][0]

    pusch_configs = pusch_config(input_file)

    cfo_ta_estimator = CfoTaEstimator(
        num_rx_ant=num_rx_ant,
        mu=mu,
        cuda_stream=cuda_stream
    )

    cfo_hz, ta = cfo_ta_estimator.estimate(
        channel_est=[ch_est],
        pusch_configs=pusch_configs
    )
    cfo_est = cfo_ta_estimator.cfo_est[0]

    ref_cfo_hz = np.array(input_file["reference_cfoEstHzPerUe"]).T[:, 0]
    ref_ta = np.array(input_file["reference_taEstMicroSecPerUe"]).T[:, 0]
    ref_cfo_est = np.array(input_file["reference_cfoEst0"]).T
    ref_cfo_est = ref_cfo_est["re"] + 1j * ref_cfo_est["im"]

    assert np.allclose(cfo_hz, ref_cfo_hz, rtol=0.01)
    assert np.allclose(ta, ref_ta, rtol=0.1)

    # Reference value exists only for one UE.
    assert np.allclose(cfo_est[:, :1], ref_cfo_est, rtol=0.01)
