# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - PUSCH receiver."""
from typing import List
from typing import Tuple

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.params import get_pusch_stat_prms
from aerial.phy5g.params import get_pusch_dyn_prms
from aerial.phy5g.params import pusch_config_to_dyn_prms
from aerial.phy5g.params import PuschConfig
from aerial.phy5g.params import get_pusch_dyn_prms_phase_2
from aerial.phy5g.types import PuschEqCoefAlgoType
from aerial.phy5g.types import PuschLdpcKernelLaunch

# Constant definitions.
NUM_PRB_MAX = 273


class PuschRx:
    """PUSCH receiver pipeline.

    This class implements the whole PUSCH reception pipeline from the received OFDM
    post-FFT symbols to the received transport block (along with CRC check).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_ul_bwp: int = NUM_PRB_MAX,
        num_dl_bwp: int = NUM_PRB_MAX,
        mu: int = 1,
        enable_cfo_correction: int = 0,
        enable_to_estimation: int = 0,
        enable_pusch_tdi: int = 0,
        eq_coeff_algo: int = 1,
        enable_per_prg_chest: int = 0,
        enable_ul_rx_bf: int = 0,
        ldpc_kernel_launch: PuschLdpcKernelLaunch = PuschLdpcKernelLaunch.PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH  # noqa: E501 # pylint: disable=line-too-long
    ) -> None:
        """Initialize PuschRx.

        Args:
            cell_id (int): Physical cell ID.
            num_rx_ant (int): Number of receive antennas.
            num_tx_ant (int): Number of transmit antennas.
            num_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
                Default: 273.
            num_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
                Default: 273.
            mu (int): Numerology. Values in [0, 3]. Default: 1.
            enable_cfo_correction (int): Enable/disable CFO correction:

                - 0: Disable (default).
                - 1: Enable.

            enable_to_estimation (int): Enable/disable time offset estimation:

                - 0: Disable (default).
                - 1: Enable.

            enable_pusch_tdi (int): Time domain interpolation on PUSCH.

                - 0: Disable (default).
                - 1: Enable.

            eq_coeff_algo (int): Algorithm for equalizer coefficient computation.

                - 0 - ZF.
                - 1 - MMSE (default).
                - 2 - MMSE-IRC.

            enable_per_prg_chest (int): Enable/disable PUSCH per-PRG channel estimation.

                - 0: Disable (default).
                - 1: Enable.

            enable_ul_rx_bf (int): Enable/disable beamforming for PUSCH.

                - 0: Disable (default).
                - 1: Enable.

            ldpc_kernel_launch (PuschLdpcKernelLaunch): LDPC kernel launch method.
        """
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant

        # TODO: Enable arbitrary user-defined stat_prms, e.g. by exposing
        # more parameters through this function interface.
        self.pusch_rx_stat_prms = get_pusch_stat_prms(
            cell_id=cell_id,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            num_prb_ul_bwp=num_ul_bwp,
            num_prb_dl_bwp=num_dl_bwp,
            mu=mu,
            enable_cfo_correction=enable_cfo_correction,
            enable_to_estimation=enable_to_estimation,
            enable_pusch_tdi=enable_pusch_tdi,
            enable_per_prg_chest=enable_per_prg_chest,
            enable_ul_rx_bf=enable_ul_rx_bf,
            eq_coeff_algo=PuschEqCoefAlgoType(eq_coeff_algo),
            ldpc_kernel_launch=ldpc_kernel_launch
        )
        self.stream = check_cuda_errors(cudart.cudaStreamCreate())
        # pylint: disable=no-member
        self.pusch_pipeline = pycuphy.PuschPipeline(self.pusch_rx_stat_prms, self.stream)

    def run(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        rx_slot: np.ndarray,
        slot: int = 0,

        pusch_configs: List[PuschConfig] = None,

        # UE group parameters.
        num_ues: int = 1,
        num_dmrs_cdm_grps_no_data: int = 2,
        dmrs_scrm_id: int = 41,
        start_prb: int = 0,
        num_prbs: int = 273,
        prg_size: int = 1,
        num_ul_streams: int = 1,
        dmrs_syms: List[int] = None,
        dmrs_max_len: int = 2,
        dmrs_add_ln_pos: int = 1,
        start_sym: int = 2,
        num_symbols: int = 12,

        # UE parameters.
        scids: List[int] = None,
        layers: List[int] = None,
        dmrs_ports: List[int] = None,
        rntis: List[int] = None,
        data_scids: List[int] = None,

        # CW parameters.
        mcs_tables: List[int] = None,
        mcs_indices: List[int] = None,
        code_rates: List[int] = None,
        mod_orders: List[int] = None,
        tb_sizes: List[int] = None,
        rvs: List[int] = None,
        ndis: List[int] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run PUSCH Rx.

        This runs the cuPHY PUSCH receiver pipeline based on the given parameters. Multiple
        UE groups are supported if the `PuschConfig` based API is used. Otherwise, the pipeline
        gets run only for a single UE group sharing the same time-frequency resources, i.e.
        having the same PRB allocation, and the same start symbol and number of allocated symbols.
        In this case default values get filled for the parameters that are not given.

        Args:
            rx_slot: A tensor representing the receive slot buffer of the cell.
            slot (int): Slot number.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, the other
                arguments will be used (default values are used for the parameters that are not
                given). Only one UE group is supported in that case.
            num_ues (int): Number of UEs in the UE group.
            num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
                [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
            dmrs_scrm_id (int): DMRS scrambling ID.
            start_prb (int): Start PRB index of the UE group allocation.
            num_prbs (int): Number of allocated PRBs for the UE group.
            prg_size (int): The Size of PRG in PRB for PUSCH per-PRG channel estimation.
            nUplinkStreams (int): The number of active streams for this PUSCH.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol.
            dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
                single-symbol DMRS or single- or double-symbol DMRS.
            dmrs_add_ln_pos (int): Number of additional DMRS positions.
            start_sym (int): Start OFDM symbol index for the UE group allocation.
            num_symbols (int): Number of symbols in the UE group allocation.
            scids (List[int]): DMRS sequence initialization for each UE
                [TS38.211, sec 7.4.1.1.2].
            layers (List[int]): Number of layers for each UE.
            dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
                SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
                indicates whether the corresponding DMRS port index is used.
            rntis (List[int]) RNTI for each UE.
            data_scids (List[int]): Data scrambling IDs for each UE, more precisely
                `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
            mcs_tables (List[int]): MCS table to use for each UE (see TS 38.214).
            mcs_indices (List[int]): MCS indices for each UE.
            code_rates (List[float]): Code rate for each UE. This is the number of information bits
                per 1024 coded bits.
            mod_orders (List[int]): Modulation order for each UE.
            tb_sizes (List[int]): TB size in bytes for each UE.
            rvs (List[int]): Redundancy versions for each UE.
            ndis (List[int]): New data indicator per UE.

        Returns:
            np.ndarray, List[np.ndarray]: A tuple containing:

            - *np.ndarray*: Transport block CRCs.
            - *List[np.ndarray]*: Transport blocks, one per UE, without CRC.
        """
        # Set dynamic prms.
        if pusch_configs is not None:
            pusch_rx_dyn_params = pusch_config_to_dyn_prms(
                cuda_stream=self.stream,
                rx_data=[rx_slot],
                slot=slot,
                pusch_configs=pusch_configs
            )
            tb_sizes = []
            num_ues = 0
            for pusch_config in pusch_configs:
                tb_sizes += [ue_config.tb_size for ue_config in pusch_config.ue_configs]
                num_ues += len(pusch_config.ue_configs)
        else:
            pusch_rx_dyn_params = get_pusch_dyn_prms(
                cuda_stream=self.stream,
                rx_data=[rx_slot],
                num_ues=num_ues,
                slot=slot,
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
                scids=scids,
                layers=layers,
                dmrs_ports=dmrs_ports,
                rntis=rntis,
                data_scids=data_scids,
                mcs_tables=mcs_tables,
                mcs_indices=mcs_indices,
                target_code_rates=[c * 10 for c in code_rates],
                mod_orders=mod_orders,
                tb_sizes=tb_sizes,
                rvs=rvs,
                ndis=ndis
            )

        # Run setup phase 1.
        self.pusch_pipeline.setup_pusch_rx(pusch_rx_dyn_params)

        # Run setup phase 2.
        harq_buffers = []
        for ue_idx in range(num_ues):
            harq_buffer_size = pusch_rx_dyn_params.dataOut.harqBufferSizeInBytes[ue_idx]

            # TODO: Move this out of the real-time pipeline.
            harq_buffer = check_cuda_errors(cudart.cudaMalloc(harq_buffer_size))
            check_cuda_errors(
                cudart.cudaMemsetAsync(
                    harq_buffer, 0, harq_buffer_size * 1, self.stream
                )
            )
            check_cuda_errors(cudart.cudaStreamSynchronize(self.stream))
            harq_buffers.append(harq_buffer)

        pusch_rx_dyn_params = get_pusch_dyn_prms_phase_2(
            pusch_rx_dyn_params, harq_buffers
        )
        self.pusch_pipeline.setup_pusch_rx(pusch_rx_dyn_params)

        # Run pipeline.
        self.pusch_pipeline.run_pusch_rx()

        # Fetch outputs.
        # Please note that not all PUSCH features are propagated through to pyaerial and not
        # all PuschDataOut processing results are checked, e.g., pCbCrcs are not.
        tb_crcs = pusch_rx_dyn_params.dataOut.tbCrcs
        tb_payloads = pusch_rx_dyn_params.dataOut.tbPayloads
        tot_num_tb_bytes = pusch_rx_dyn_params.dataOut.totNumPayloadBytes
        start_offsets = list(pusch_rx_dyn_params.dataOut.startOffsetsTbPayload)
        start_offsets.append(tot_num_tb_bytes[0])
        tbs = []
        for ue_idx in range(num_ues):
            # Remove CRC and padding bytes (cuPHY aligns the output to 4-byte boundaries).
            tb = tb_payloads[start_offsets[ue_idx] : start_offsets[ue_idx + 1]]
            tb = tb[:tb_sizes[ue_idx]]
            tbs.append(tb)

        # TODO: Move this out of the real-time pipeline.
        for ue_idx in range(num_ues):
            check_cuda_errors(cudart.cudaFree(harq_buffers[ue_idx]))

        return tb_crcs, tbs
