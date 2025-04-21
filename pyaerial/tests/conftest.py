# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""The conftest.py file for pytest. Contains common fixtures etc."""
import gc
import os
import pytest

from cuda import cudart
import numpy as np

from aerial.util.cuda import check_cuda_errors
from aerial.util.fapi import dmrs_fapi_to_bit_array
from aerial.phy5g.pdsch import PdschTx
from aerial.phy5g.pusch import PuschRx
from aerial.phy5g.params import PuschConfig
from aerial.phy5g.params import PuschUeConfig


@pytest.fixture(scope="function")
def clear_session():
    """Force garbage collection at the end of a test."""
    # Yield and let test run to completion.
    yield
    gc.collect()


def pytest_configure():
    """Configure pytest."""
    pytest.TEST_VECTOR_DIR = os.environ.get(
        "TEST_VECTOR_DIR",
        "/mnt/cicd_tvs/develop/GPU_test_input/"
    )


@pytest.fixture(name="cuda_stream", scope="module")
def fixture_cuda_stream():
    """A fixture for setting up CUDA and creating a CUDA stream."""
    # TODO: Figure out how to set the GPU in CICD.
    gpu_id = 0
    cudart.cudaSetDevice(gpu_id)

    # Allocate a CUDA stream.
    cu_stream = check_cuda_errors(cudart.cudaStreamCreate())
    check_cuda_errors(cudart.cudaStreamSynchronize(cu_stream))

    # Let the test run and the destroy the CUDA stream.
    return cu_stream


@pytest.fixture(name="pdsch_tx", scope="function")
def fixture_pdsch_tx():
    """Fixture for creating a PdschTx object."""
    pdsch_tx = PdschTx(
        cell_id=41,
        num_rx_ant=4,
        num_tx_ant=4,
    )
    return pdsch_tx


@pytest.fixture(name="pusch_rx", scope="module")
def fixture_pusch_rx():
    """Fixture for creating a PuschRx object."""
    pusch_rx = PuschRx(
        cell_id=41,
        num_rx_ant=4,
        num_tx_ant=4,
    )
    return pusch_rx


@pytest.fixture(name="pusch_config", scope="module")
def fixture_pusch_config():
    """Fixture that returns a function for getting PuschConfig out of a test vector file."""
    def _pusch_config(test_vector_file):

        num_ue_grps = len(test_vector_file["ueGrp_pars"])
        num_ues = len(test_vector_file["tb_pars"])

        pusch_configs = []
        pusch_ue_configs = [[] for _ in range(num_ue_grps)]

        tb_pars = test_vector_file["tb_pars"]
        for ue_idx in range(num_ues):
            scid = tb_pars["nSCID"][ue_idx]
            num_layers = tb_pars["numLayers"][ue_idx]
            dmrs_ports = tb_pars["dmrsPortBmsk"][ue_idx]
            rnti = tb_pars["nRnti"][ue_idx]
            data_scid = tb_pars["dataScramId"][ue_idx]
            mcs_table = tb_pars["mcsTableIndex"][ue_idx]
            mcs_index = tb_pars["mcsIndex"][ue_idx]
            code_rate = tb_pars["targetCodeRate"][ue_idx]
            mod_order = tb_pars["qamModOrder"][ue_idx]
            tb_size = tb_pars["nTbByte"][ue_idx]
            rv = tb_pars["rv"][ue_idx]
            ndi = tb_pars["ndi"][ue_idx]
            ue_grp_idx = tb_pars["userGroupIndex"][ue_idx]

            pusch_ue_config = PuschUeConfig(
                scid=scid,
                layers=num_layers,
                dmrs_ports=dmrs_ports,
                rnti=rnti,
                data_scid=data_scid,
                mcs_table=mcs_table,
                mcs_index=mcs_index,
                code_rate=code_rate,
                mod_order=mod_order,
                tb_size=tb_size,
                rv=rv,
                ndi=ndi,
                harq_process_id=0
            )
            pusch_ue_configs[ue_grp_idx].append(pusch_ue_config)

        for ue_grp_idx in range(num_ue_grps):
            first_ue_idx = test_vector_file["ueGrp_pars"]["UePrmIdxs"][ue_grp_idx]
            if isinstance(first_ue_idx, np.ndarray):
                first_ue_idx = first_ue_idx[0]

            num_dmrs_cdm_grps_no_data = tb_pars["numDmrsCdmGrpsNoData"][first_ue_idx]
            dmrs_scrm_id = tb_pars["dmrsScramId"][first_ue_idx]
            start_prb = test_vector_file["ueGrp_pars"]["startPrb"][ue_grp_idx]
            num_prbs = test_vector_file["ueGrp_pars"]["nPrb"][ue_grp_idx]
            prg_size = test_vector_file["ueGrp_pars"]["prgSize"][ue_grp_idx]
            num_ul_streams = test_vector_file["ueGrp_pars"]["nUplinkStreams"][ue_grp_idx]
            dmrs_sym_loc_bmsk = test_vector_file["ueGrp_pars"]["dmrsSymLocBmsk"][ue_grp_idx]
            dmrs_syms = dmrs_fapi_to_bit_array(dmrs_sym_loc_bmsk)
            dmrs_max_len = tb_pars["dmrsMaxLength"][first_ue_idx]
            dmrs_add_ln_pos = tb_pars["dmrsAddlPosition"][first_ue_idx]
            start_sym = test_vector_file["ueGrp_pars"]["StartSymbolIndex"][ue_grp_idx]
            num_symbols = test_vector_file["ueGrp_pars"]["NrOfSymbols"][ue_grp_idx]

            pusch_config = PuschConfig(
                num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
                dmrs_scrm_id=dmrs_scrm_id,
                start_prb=start_prb,
                num_prbs=num_prbs,
                prg_size=prg_size,
                num_ul_streams=num_ul_streams,
                dmrs_syms=dmrs_syms,
                dmrs_max_len=dmrs_max_len,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                start_sym=start_sym,
                num_symbols=num_symbols,
                ue_configs=pusch_ue_configs[ue_grp_idx]
            )

            pusch_configs.append(pusch_config)

        return pusch_configs

    return _pusch_config
