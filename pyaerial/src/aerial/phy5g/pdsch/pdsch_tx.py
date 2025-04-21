# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - PDSCH transmitter."""
from typing import List
from typing import Tuple

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.params import get_pdsch_stat_prms
from aerial.phy5g.params import get_pdsch_dyn_prms
from aerial.phy5g.params import CsiRsRrcDynPrms


# Constant definitions.
NUM_RE_PER_PRB = 12
NUM_PRB_MAX = 273
NUM_SYMBOLS = 14
MAX_NUM_CODE_BLOCKS = 152
MAX_DL_LAYERS = 16


class PdschTx:
    """PDSCH transmitter.

    This class implements the whole PDSCH transmission pipeline from the transmitted
    transport block to the transmitted frequency-domain symbols.
    """
    def __init__(
        self,
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_ul_bwp: int = NUM_PRB_MAX,
        num_dl_bwp: int = NUM_PRB_MAX,
        mu: int = 1,
    ) -> None:
        """Initialize PdschTx.

        Args:
            cell_id (int): Physical cell ID.
            num_rx_ant (int): Number of receive antennas.
            num_tx_ant (int): Number of transmit antennas.
            num_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
                Default: 273.
            num_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
                Default: 273.
            mu (int): Numerology. Values in [0, 3]. Default: 1.
        """
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant

        # TODO: Enable arbitrary user-defined stat_prms.
        pdsch_tx_stat_prms = get_pdsch_stat_prms(
            cell_id=cell_id,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            num_prb_ul_bwp=num_ul_bwp,
            num_prb_dl_bwp=num_dl_bwp,
            mu=mu,
        )
        # pylint: disable=no-member
        self.output_dims = [num_dl_bwp * NUM_RE_PER_PRB, NUM_SYMBOLS, MAX_DL_LAYERS]
        self.pdsch_pipeline = pycuphy.PdschPipeline(pdsch_tx_stat_prms)
        self.stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.device_tx_tensor_mem = check_cuda_errors(
            cudart.cudaMalloc(NUM_PRB_MAX * NUM_RE_PER_PRB * NUM_SYMBOLS * MAX_DL_LAYERS * 4)
        )
        self.host_tx_tensor_mem = check_cuda_errors(
            cudart.cudaMallocHost(NUM_PRB_MAX * NUM_RE_PER_PRB * NUM_SYMBOLS * MAX_DL_LAYERS * 8)
        )
        self.host_ldpc_output_mem = check_cuda_errors(
            cudart.cudaMallocHost(66 * 384 * MAX_NUM_CODE_BLOCKS * 4)
        )

    def run(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        tb_inputs: List[np.ndarray],
        num_ues: int,
        slot: int,

        # UE group parameters.
        num_dmrs_cdm_grps_no_data: int = 2,
        dmrs_scrm_id: int = 41,
        resource_alloc: int = 1,
        prb_bitmap: List[int] = None,
        start_prb: int = 0,
        num_prbs: int = 273,
        dmrs_syms: List[int] = None,
        start_sym: int = 2,
        num_symbols: int = 12,

        # UE parameters.
        scids: List[int] = None,
        layers: List[int] = None,
        dmrs_ports: List[int] = None,
        bwp_starts: List[int] = None,
        ref_points: List[int] = None,
        rntis: List[int] = None,
        data_scids: List[int] = None,
        precoding_matrices: List[np.ndarray] = None,

        # CW parameters.
        mcs_tables: List[int] = None,
        mcs_indices: List[int] = None,
        code_rates: List[int] = None,
        mod_orders: List[int] = None,
        rvs: List[int] = None,
        num_prb_lbrms: List[int] = None,
        max_layers: List[int] = None,
        max_qms: List[int] = None,

        # CSI-RS parameters.
        csirs_rrc_dyn_prms: List[CsiRsRrcDynPrms] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run PDSCH transmission.

        Set dynamic PDSCH parameters and call cuPHY to run the PDSCH transmission.

        Args:
            tb_inputs (List[np.ndarray]): Transport blocks in bytes for each UE.
            num_ues (int): Number of UEs.
            slot (int): Slot number.
            num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
                [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
            dmrs_scrm_id (int): Downlink DMRS scrambling ID.
            resource_alloc (int): Resource allocation type.
            prb_bitmap (List[int]): Array of bytes indicating bitmask for allocated RBs.
            start_prb (int): Start PRB index for the UE group.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol.
            start_sym (int): Start OFDM symbol index of the UE group allocation.
            num_symbols (int): Number of symbols in the allocation, starting from
                `start_sym`.
            scids (List[int]): DMRS sequence initialization for each UE
                [TS38.211, sec 7.4.1.1.2].
            layers (List[int]): Number of layers for each UE.
            dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
                SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
                indicates whether the corresponding DMRS port index is used.
            bwp_starts (List[int]): Bandwidth part start (PRB number starting from 0).
                Used only if reference point is 1.
            ref_points (List[int]): DMRS reference point per UE. Value 0 or 1.
            rntis (List[int]) RNTI for each UE.
            data_scids (List[int]): Data scrambling IDs for each UE, more precisely
                `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
            precoding_matrices (List[np.ndarray]): Precoding matrices, one per UE.
                The shape of each precoding matrix is number of layers x number of Tx antennas.
                If set to None, precoding is disabled.
            mcs_tables (List[int]): MCS table per UE.
            mcs_indices (List[int]): MCS index per UE.
            code_rates (List[int]): Code rate for each UE in 3GPP format,
                i.e. code rate x 1024.
            mod_orders (List[int]): Modulation order for each UE.
            rvs (List[int]): Redundancy version per UE (default: 0 for each UE).
            num_prb_lbrms (List[int]): Number of PRBs used for LBRM TB size computation.
                Possible values: {32, 66, 107, 135, 162, 217, 273}.
            max_layers (List[int]): Number of layers used for LBRM TB size computation (at most 4).
            max_qms (List[int]): Modulation order used for LBRM TB size computation. Value: 6 or 8.
            csirs_rrc_dyn_prms (List[CsiRsRrcDynPrms]): List of CSI-RS RRC dynamic parameters, see
                `CsiRsRrcDynPrms`. Note that no CSI-RS symbols get written, this is only to make
                sure that PDSCH does not get mapped to the CSI-RS resource elements.

        Returns:
            np.ndarray, np.ndarray: A tuple containing:

            - *np.ndarray*: Transmitted OFDM symbols in a frequency x time x antenna tensor.
            - *np.ndarray*: Coded bits in a num_codewords x num_bits_per_codeword tensor.
        """
        if precoding_matrices is not None:
            precoding_matrices = [np.ascontiguousarray(m) for m in precoding_matrices]

        # Reset output buffer.
        check_cuda_errors(
            cudart.cudaMemsetAsync(self.device_tx_tensor_mem,
                                   0,
                                   NUM_PRB_MAX * NUM_RE_PER_PRB * NUM_SYMBOLS * MAX_DL_LAYERS * 4,
                                   self.stream)
        )
        check_cuda_errors(cudart.cudaStreamSynchronize(self.stream))

        # Create the dynamic params structure. Default parameters inserted for those
        # that are not given.
        pdsch_tx_dyn_prms = get_pdsch_dyn_prms(
            cuda_stream=self.stream,
            device_tx_tensor_mem=self.device_tx_tensor_mem,
            tb_inputs=tb_inputs,
            num_ues=num_ues,
            slot=slot,

            # UE group parameters.
            num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
            dmrs_scrm_id=dmrs_scrm_id,
            resource_alloc=resource_alloc,
            prb_bitmap=prb_bitmap,
            start_prb=start_prb,
            num_prbs=num_prbs,
            dmrs_syms=dmrs_syms,
            start_sym=start_sym,
            num_symbols=num_symbols,

            # UE parameters.
            scids=scids,
            layers=layers,
            dmrs_ports=dmrs_ports,
            bwp_starts=bwp_starts,
            ref_points=ref_points,
            rntis=rntis,
            data_scids=data_scids,
            precoding_matrices=precoding_matrices,

            # CW parameters.
            mcs_tables=mcs_tables,
            mcs_indices=mcs_indices,
            target_code_rates=[c * 10 for c in code_rates],
            mod_orders=mod_orders,
            rvs=rvs,
            num_prb_lbrms=num_prb_lbrms,
            max_layers=max_layers,
            max_qms=max_qms,

            # CSI-RS parameters
            csirs_rrc_dyn_prms=csirs_rrc_dyn_prms
        )
        self.pdsch_pipeline.setup_pdsch_tx(pdsch_tx_dyn_prms)
        check_cuda_errors(cudart.cudaStreamSynchronize(self.stream))
        self.pdsch_pipeline.run_pdsch_tx()
        check_cuda_errors(cudart.cudaStreamSynchronize(self.stream))

        # Get the coded bits i.e. the LDPC output.
        ldpc_output = self.pdsch_pipeline.get_ldpc_output(0, 0, self.host_ldpc_output_mem)

        # Get the transmitted OFDM REs.
        tx_slot = pycuphy.device_to_numpy(  # pylint: disable=no-member
            self.device_tx_tensor_mem,
            self.host_tx_tensor_mem,
            self.output_dims,
            self.stream
        )
        tx_slot = PdschTx.cuphy_to_tx(tx_slot, num_ues, dmrs_ports, scids, precoding_matrices)
        return tx_slot, ldpc_output

    @classmethod
    def cuphy_to_tx(
            cls,
            tx_slot: np.ndarray,
            num_ues: int,
            dmrs_ports: List[int],
            scids: List[int],
            precoding_matrices: List[np.ndarray] = None) -> np.ndarray:
        """Map cuPHY outputs to Tx antenna ports.

        Args:
            tx_slot (numpy.ndarray): Transmit buffer from cuPHY.
            num_ues (int): Number of UEs.
            dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
                SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
                indicates whether the corresponding DMRS port index is used.
            scids (List[int]): DMRS sequence initialization for each UE [TS38.211, sec 7.4.1.1.2].
            precoding_matrices (List[np.ndarray]): Precoding matrices, one per UE.
                The shape of each precoding matrix is number of layers x number of Tx antennas.
                If set to None, precoding is disabled.

        Returns:
            np.ndarray: Transmitted OFDM symbols in a frequency x time x antenna tensor.
        """
        indices = []
        for ue in range(num_ues):
            if precoding_matrices is None or precoding_matrices[ue].size == 0:
                dmrs_port_indices = np.where(np.flipud(np.unpackbits(np.uint8(dmrs_ports[ue]))))[0]
                indices += list(dmrs_port_indices + 8 * scids[ue])
            else:
                indices += list(range(precoding_matrices[ue].shape[1]))
        indices = list(set(indices))

        return tx_slot[:, :, indices]

    def __del__(self) -> None:
        """Destructor."""
        check_cuda_errors(cudart.cudaFree(self.device_tx_tensor_mem))
        check_cuda_errors(cudart.cudaFreeHost(self.host_tx_tensor_mem))
        check_cuda_errors(cudart.cudaFreeHost(self.host_ldpc_output_mem))
