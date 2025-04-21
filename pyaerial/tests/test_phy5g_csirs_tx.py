# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Tests for csirs_tx.py."""
import glob
import pytest
from pytest import TEST_VECTOR_DIR

import h5py as h5
import numpy as np

from aerial.phy5g.types import CsiRsRrcDynPrms
from aerial.phy5g.types import CsiRsCellDynPrms
from aerial.phy5g.types import CsiRsPmwOneLayer
from aerial.phy5g.pdsch import CsiRsTx


# Test vector numbers.
test_case_numbers = list(range(4001, 4063))
test_case_numbers += [4101, 4102, 4103]
test_case_numbers += list(range(4201, 4223))
test_case_numbers += list(range(4801, 4808))
test_case_numbers += list(range(4901, 4906))


# pylint: disable=too-many-locals
@pytest.mark.parametrize(
    "test_case_number",
    test_case_numbers,
    ids=[int(test_case_number) for test_case_number in test_case_numbers]
)
def test_csirs_tx_run(test_case_number):
    """Test running CSI-RS Tx against Aerial test vectors."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_CSIRS_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    csirs_prms_list = input_file["CsirsParamsList"]
    csirs_rrc_prms = []
    for csirs_prms in csirs_prms_list:
        csirs_rrc_prms.append(CsiRsRrcDynPrms(
            start_prb=np.uint16(csirs_prms["StartRB"]),
            num_prb=np.uint16(csirs_prms["NrOfRBs"]),
            prb_bitmap=list(map(int, format(csirs_prms["FreqDomain"], "016b"))),
            row=np.uint8(csirs_prms["Row"]),
            symb_L0=np.uint8(csirs_prms["SymbL0"]),
            symb_L1=np.uint8(csirs_prms["SymbL1"]),
            freq_density=np.uint8(csirs_prms["FreqDensity"]),
            scramb_id=np.uint16(csirs_prms["ScrambId"]),
            idx_slot_in_frame=np.uint8(csirs_prms["idxSlotInFrame"]),
            csi_type=np.uint8(csirs_prms["CSIType"]),
            cdm_type=np.uint8(csirs_prms["CDMType"]),
            beta=csirs_prms["beta"],
            enable_precoding=np.uint8(csirs_prms["enablePrcdBf"]),
            pmw_prm_idx=np.uint16(0)
        ))
    csirs_cell_dyn_prms = CsiRsCellDynPrms(rrc_dyn_prms=csirs_rrc_prms)

    ref_tx_buffer = np.array(input_file["X_tf"])["re"] + 1j * np.array(input_file["X_tf"])["im"]
    ref_tx_buffer = np.ascontiguousarray(ref_tx_buffer.T)

    csirs_tx = CsiRsTx(num_prb_dl_bwp=[ref_tx_buffer.shape[0] // 12])

    csirs_pmw = np.array(input_file["Csirs_PM_W0"])
    csirs_pmw = csirs_pmw["re"] + 1j * csirs_pmw["im"]

    if csirs_pmw.size:
        precoding_matrices = [CsiRsPmwOneLayer(precoding_matrix=csirs_pmw,
                                               num_ports=csirs_pmw.shape[1])]
    else:
        precoding_matrices = None

    tx_buffer = np.zeros(ref_tx_buffer.shape, dtype=np.complex64)
    tx_buffer = csirs_tx.run(
        csirs_cell_dyn_prms=[csirs_cell_dyn_prms],
        precoding_matrices=precoding_matrices,
        tx_buffers=[tx_buffer]
    )[0]

    assert np.allclose(tx_buffer, ref_tx_buffer, rtol=0.001)
