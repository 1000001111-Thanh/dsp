# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial - cuPHY API parameter utilities."""
from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from aerial.phy5g.types import (
    PdschStatPrms, CellStatPrm, PdschCellDynPrm, PdschUeGrpPrm,
    PdschUePrm, PdschCwPrm, PdschCellGrpDynPrm, PdschDataIn,
    CuPHYTensor, PdschDataOut, PdschDynPrms, DataType,
    CuPHYTracker, PmW, PdschDbgPrms,
    PuschStatPrms, PuschStatDbgPrms, PuschCellDynPrm, PuschUeGrpPrm,
    PuschDmrsPrm, PuschUePrm, PuschCellGrpDynPrm,
    PuschDataIn, PuschDataOut, PuschDataInOut,
    PuschDynPrms, PuschDynDbgPrms, PuschSetupPhase,
    PuschEqCoefAlgoType, PuschLdpcKernelLaunch,
    LdpcMaxItrAlgoType, PuschStatusOut, PuschStatusType,
    PuschChEstAlgoType, PuschWorkCancelMode, CsiRsRrcDynPrms
)
from aerial.util.fapi import dmrs_bit_array_to_fapi
from aerial.phy5g.chest_filters import (
    w_freq_array, w_freq4_array, w_freq_small_array, shift_seq_array,
    unshift_seq_array, shift_seq4_array, unshift_seq4_array
)

# Constant definitions.
NUM_RE_PER_PRB = 12
NUM_PRB_MAX = 273
NUM_SYMBOLS = 14


@dataclass
class PuschUeConfig:
    """A class holding all dynamic PUSCH parameters for a single slot, single UE.

    Args:
        scid (int): DMRS sequence initialization [TS38.211, sec 7.4.1.1.2].
        layers (int): Number of layers.
        dmrs_ports (int): Allocated DMRS ports.
        rnti (int): The 16-bit RNTI value of the UE.
        data_scid (List[int]): Data scrambling ID, more precisely `dataScramblingIdentityPdsch`
            [TS38.211, sec 7.3.1.1].
        mcs_table (int): MCS table to use (see TS 38.214).
        mcs_index (int): MCS index to use.
        code_rate (int): Code rate, expressed as the number of information
            bits per 1024 coded bits expressed in 0.1 bit units.
        mod_order (int): Modulation order.
        tb_size (int): TB size in bytes.
        rv (List[int]): Redundancy version.
        ndi (List[int]): New data indicator.
    """
    scid: int = 0
    layers: int = 1
    dmrs_ports: int = 1
    rnti: int = 1
    data_scid: int = 41
    mcs_table: int = 0
    mcs_index: int = 0
    code_rate: int = 1930
    mod_order: int = 2
    tb_size: int = 96321
    rv: int = 0
    ndi: int = 1
    harq_process_id: int = 0


@dataclass
class PuschConfig:
    """A class holding all dynamic PUSCH parameters for a single slot, single UE group.

    Args:
        num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
            [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
        dmrs_scrm_id (int): DMRS scrambling ID.
        start_prb (int): Start PRB index of the UE group allocation.
        num_prbs (int): Number of allocated PRBs for the UE group.
        prg_size (int): The Size of PRG in PRB for PUSCH per-PRG channel estimation.
        num_ul_streams (int): The number of active streams for this PUSCH.
        dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
            single-symbol DMRS or single- or double-symbol DMRS. Note that this needs to be
            consistent with `dmrs_syms`.
        dmrs_add_ln_pos (int): Number of additional DMRS positions.  Note that this needs to be
            consistent with `dmrs_syms`.
        start_sym (int): Start OFDM symbol index for the UE group allocation.
        num_symbols (int): Number of symbols in the UE group allocation.
    """
    # UE parameters.
    ue_configs: List[PuschUeConfig]

    # UE group parameters.
    num_dmrs_cdm_grps_no_data: int = 2
    dmrs_scrm_id: int = 41
    start_prb: int = 0
    num_prbs: int = 273
    prg_size: int = 1
    num_ul_streams: int = 1
    dmrs_syms: List[int] = \
        field(default_factory=lambda: [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    dmrs_max_len: int = 2
    dmrs_add_ln_pos: int = 1
    start_sym: int = 2
    num_symbols: int = 12


def get_pdsch_stat_prms(
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_rx_ant_srs: Optional[int] = None,
        num_prb_ul_bwp: int = NUM_PRB_MAX,
        num_prb_dl_bwp: int = NUM_PRB_MAX,
        mu: int = 1) -> PdschStatPrms:
    """Get a simple PdschStatPrms object based on given parameters.

    Args:
        cell_id (int): Physical cell ID.
        num_rx_ant (int): Number of receive antennas.
        num_tx_ant (int): Number of transmit antennas.
        num_rx_ant_srs (int): Number of receive antennas for SRS.
        num_prb_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
            Default: 273.
        num_prb_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
            Default: 273.
        mu (int): Numerology. Values in [0, 3]. Default: 1.

    Returns:
        PdschStatPrms: The PdschStatPrms object.
    """
    num_rx_ant_srs = num_rx_ant_srs or num_rx_ant
    cell_stat_prm = CellStatPrm(
        phyCellId=np.uint16(cell_id),
        nRxAnt=np.uint16(num_rx_ant),
        nRxAntSrs=np.uint16(num_rx_ant_srs),
        nTxAnt=np.uint16(num_tx_ant),
        nPrbUlBwp=np.uint16(num_prb_ul_bwp),
        nPrbDlBwp=np.uint16(num_prb_dl_bwp),
        mu=np.uint8(mu)
    )

    cuphy_tracker = CuPHYTracker(
        memoryFootprint=[]  # Not used by pycuphy code.
    )

    dbg_params = PdschDbgPrms(
        cfgFilename=None,
        checkTbSize=np.uint8(1),
        refCheck=False,
        cfgIdenticalLdpcEncCfgs=False
    )

    pdsch_tx_stat_prms = PdschStatPrms(
        outInfo=[cuphy_tracker],
        cellStatPrms=[cell_stat_prm],
        dbg=[dbg_params],
        read_TB_CRC=False,
        full_slot_processing=True,
        stream_priority=0,
        nMaxCellsPerSlot=np.uint16(1),
        nMaxUesPerCellGroup=np.uint16(0),
        nMaxCBsPerTB=np.uint16(0),
        nMaxPrb=np.uint16(0)
    )

    return pdsch_tx_stat_prms


def get_pdsch_dyn_prms(  # noqa: C901 pylint: disable=too-many-arguments, too-many-locals
        cuda_stream: int,
        device_tx_tensor_mem: np.uint64,
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
        target_code_rates: List[int] = None,
        mod_orders: List[int] = None,
        rvs: List[int] = None,
        num_prb_lbrms: List[int] = None,
        max_layers: List[int] = None,
        max_qms: List[int] = None,

        # CSI-RS parameters.
        csirs_rrc_dyn_prms: List[CsiRsRrcDynPrms] = None) -> PdschDynPrms:
    """Get a simple PdschDynPrms object based on given parameters.

    Note: This creates a simple case serving most use cases. However, it does
    lack a number of features, for example this is one cell, one UE group only.
    In case one wants to use more advanced features of cuPHY through the Python
    bindings, the needed parameter objects need to be manually built.

    Default values are added for the parameters not given.

    Args:
        cuda_stream (int): CUDA stream on which pipeline is launched.
        device_tx_tensor_mem (int): Raw pointer to the tensor buffer.
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
        target_code_rates (List[int]): Code rate for each UE in SCF FAPI format,
            i.e. code rate x 1024 x 10.
        mod_orders (List[int]): Modulation order for each UE.
        rvs (List[int]): Redundancy version per UE (default: 0 for each UE).
        num_prb_lbrms (List[int]): Number of PRBs used for LBRM TB size computation.
            Possible values: {32, 66, 107, 135, 162, 217, 273}.
        max_layers (List[int]): Number of layers used for LBRM TB size computation (at most 4).
        max_qms (List[int]): Modulation order used for LBRM TB size computation. Value: 6 or 8.
        csirs_rrc_dyn_prms (List[CsiRsRrcDynPrms]): List of CSI-RS RRC dynamic parameters, see
            `CsiRsRrcDynPrms`. Note that no CSI-RS symbols get written, this is only to make sure
            that PDSCH does not get mapped to the CSI-RS resource elements.

    Returns:
        PdschDynPrms: The PdschDynPrms object.
    """
    # Set the default values.

    if prb_bitmap is None:
        prb_bitmap = 36 * [0, ]
    if dmrs_syms is None:
        dmrs_syms = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    if scids is None:
        valid_scids = [0, 0, 1, 1, 0, 0, 1, 1]
        scids = valid_scids[:num_ues]
    if layers is None:
        layers = list(num_ues * [1, ])
    if dmrs_ports is None:
        valid_dmrs_ports = [1, 4, 2, 8, 16, 64, 32, 128]
        dmrs_ports = valid_dmrs_ports[:num_ues]
    if bwp_starts is None:
        bwp_starts = list(num_ues * [0, ])
    if ref_points is None:
        ref_points = list(num_ues * [0, ])
    if rntis is None:
        rntis = list(np.arange(1, num_ues + 1))
    if data_scids is None:
        data_scids = list(np.arange(1, num_ues + 1))

    if mcs_tables is None:
        mcs_tables = list(num_ues * [0, ])
    if mcs_indices is None:
        mcs_indices = list(num_ues * [0, ])
    if target_code_rates is None:
        target_code_rates = list(num_ues * [1930, ])
    if mod_orders is None:
        mod_orders = list(num_ues * [2, ])
    if rvs is None:
        rvs = list(num_ues * [0, ])
    if num_prb_lbrms is None:
        num_prb_lbrms = list(num_ues * [273, ])
    if max_layers is None:
        max_layers = list(num_ues * [4, ])
    if max_qms is None:
        max_qms = list(num_ues * [8, ])
    num_csi_prms = 0
    if csirs_rrc_dyn_prms is not None:
        num_csi_prms = len(csirs_rrc_dyn_prms)

    cell_dyn_prm = PdschCellDynPrm(
        nCsiRsPrms=np.uint16(num_csi_prms),
        csiRsPrmsOffset=np.uint16(0),
        cellPrmStatIdx=np.uint16(0),
        cellPrmDynIdx=np.uint16(0),
        slotNum=np.uint16(slot),
        pdschStartSym=np.uint8(0),
        nPdschSym=np.uint8(0),
        dmrsSymLocBmsk=np.uint16(0),
        testModel=np.uint8(0)
    )

    ue_grp_prm = PdschUeGrpPrm(
        cellPrmIdx=0,
        nDmrsCdmGrpsNoData=np.uint8(num_dmrs_cdm_grps_no_data),
        dmrsScrmId=np.uint16(dmrs_scrm_id),
        resourceAlloc=np.uint8(resource_alloc),
        rbBitmap=list(np.array(prb_bitmap).astype(np.uint8)),
        startPrb=np.uint16(start_prb),
        nPrb=np.uint16(num_prbs),
        dmrsSymLocBmsk=np.uint16(dmrs_bit_array_to_fapi(dmrs_syms)),
        pdschStartSym=np.uint8(start_sym),
        nPdschSym=np.uint8(num_symbols),
        uePrmIdxs=list(np.array(range(num_ues), dtype=np.uint16))
    )

    ue_prm = []
    pmw_prms = None
    for i in range(num_ues):

        # Precoding enabled if a precoding matrix is given for the UE.
        enable_prcd_bf = False
        pmw_prm_idx = None
        if precoding_matrices is not None and precoding_matrices[i].size > 0:
            enable_prcd_bf = True
            pmw_prm_idx = i
            pmw = PmW(
                w=precoding_matrices[i],
                nPorts=np.uint8(precoding_matrices[i].shape[1])
            )
            if i == 0:
                pmw_prms = [pmw]
            else:
                pmw_prms += [pmw]  # type: ignore

        ue_prm.append(PdschUePrm(
            ueGrpPrmIdx=0,
            scid=np.uint8(scids[i]),
            nUeLayers=np.uint8(layers[i]),
            dmrsPortBmsk=np.uint16(dmrs_ports[i]),
            BWPStart=np.uint16(bwp_starts[i]),
            refPoint=np.uint8(ref_points[i]),
            beta_dmrs=np.float32(1.0),
            beta_qam=np.float32(1.0),
            rnti=np.uint16(rntis[i]),
            dataScramId=np.uint16(data_scids[i]),
            cwIdxs=[i],
            enablePrcdBf=enable_prcd_bf,
            pmwPrmIdx=pmw_prm_idx
        ))

    cw_prm = []
    tb_offset = 0
    for i in range(num_ues):
        cw_prm.append(PdschCwPrm(
            uePrmIdx=i,
            mcsTableIndex=np.uint8(mcs_tables[i]),
            mcsIndex=np.uint8(mcs_indices[i]),
            targetCodeRate=np.uint16(target_code_rates[i]),
            qamModOrder=np.uint8(mod_orders[i]),
            rv=np.uint8(rvs[i]),
            tbStartOffset=np.uint32(tb_offset),
            tbSize=np.uint32(tb_inputs[i].size),
            n_PRB_LBRM=np.uint16(num_prb_lbrms[i]),
            maxLayers=np.uint8(max_layers[i]),
            maxQm=np.uint8(max_qms[i]),
        ))
        tb_offset += tb_inputs[i].size

    cell_grp_dyn_prm = PdschCellGrpDynPrm(
        cellPrms=[cell_dyn_prm],
        ueGrpPrms=[ue_grp_prm],
        uePrms=ue_prm,  # A list.
        cwPrms=cw_prm,  # A list.
        csiRsPrms=csirs_rrc_dyn_prms,
        pmwPrms=pmw_prms
    )

    data_in = PdschDataIn(
        tbInput=[np.concatenate(tb_inputs)]
    )

    cuphy_tensor = CuPHYTensor(
        dimensions=[NUM_PRB_MAX * NUM_RE_PER_PRB, NUM_SYMBOLS, 16],
        strides=[NUM_PRB_MAX * NUM_RE_PER_PRB, NUM_SYMBOLS, 16],
        dataType=DataType.CUPHY_C_32F,
        pAddr=device_tx_tensor_mem
    )

    data_out = PdschDataOut(
        dataTx=[cuphy_tensor]
    )

    pdsch_dyn_prms = PdschDynPrms(
        cuStream=cuda_stream,
        procModeBmsk=np.uint64(4),  # Enable inter-cell batching.
        cellGrpDynPrm=cell_grp_dyn_prm,
        dataIn=data_in,
        tbCRCDataIn=None,
        dataOut=data_out
    )

    return pdsch_dyn_prms


def get_pusch_stat_prms(  # pylint: disable=too-many-arguments
        cell_id: int = 41,
        num_rx_ant: int = 4,
        num_tx_ant: int = 1,
        num_rx_ant_srs: Optional[int] = None,
        num_prb_ul_bwp: int = 273,
        num_prb_dl_bwp: int = 273,
        mu: int = 1,
        enable_cfo_correction: int = 0,
        enable_to_estimation: int = 0,
        enable_pusch_tdi: int = 0,
        ch_est_algo: PuschChEstAlgoType = PuschChEstAlgoType.PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST,  # noqa: E501 # pylint: disable=line-too-long
        enable_per_prg_chest: int = 0,
        enable_ul_rx_bf: int = 0,
        eq_coeff_algo: PuschEqCoefAlgoType = PuschEqCoefAlgoType.PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
        ldpc_kernel_launch: PuschLdpcKernelLaunch = PuschLdpcKernelLaunch.PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH,  # noqa: E501 # pylint: disable=line-too-long
        debug_file_name: Optional[str] = None) -> PuschStatPrms:
    """Get a PuschStatPrms object based on given parameters.

    Args:
        cell_id (int): Physical cell ID.
        num_rx_ant (int): Number of receive antennas.
        num_tx_ant (int): Number of transmit antennas.
        num_rx_ant_srs (int): Number of receive antennas for SRS.
        num_prb_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
            Default: 273.
        num_prb_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
            Default: 273.
        mu (int): Numerology. Values in [0, 3]. Default: 1.
        enable_cfo_correction (int): Enable/disable CFO correction:

            - 0: Disable (default).
            - 1: Enable.

        enable_to_estimation (int): Enable/disable time offset estimation:

            - 0: Disable (default).
            - 1: Enable.

        enable_pusch_tdi (int): Enable/disable time domain interpolation on PUSCH:

            - 0: Disable (default).
            - 1: Enable.

        ch_est_algo (PuschChEstAlgoType): Channel estimation algorithm.

            - 0: MMSE
            - 1: MMSE with delay estimation/compensation
            - 2: RKHS not supported by pyAerial yet
            - 3: LS channel estimation only, no interpolation, LS channel estimates get returned.

        enable_per_prg_chest (int): Enable/disable PUSCH per-PRG channel estimation.

            - 0: Disable (default).
            - 1: Enable.

        enable_ul_rx_bf (int): Enable/disable beamforming for PUSCH.

            - 0: Disable (default).
            - 1: Enable.

        eq_coeff_algo (PuschEqCoefAlgoType): Algorithm for equalizer coefficient computation.

            - 0 - ZF.
            - 1 - MMSE (default).
            - 2 - MMSE-IRC.

        ldpc_kernel_launch (PuschLdpcKernelLaunch): LDPC kernel launch method.
        debug_file_name (str): Debug dump filename. Default: None (no debugging).

    Returns:
        PuschStatPrms: The PuschStatPrms object.
    """
    num_rx_ant_srs = num_rx_ant_srs or num_rx_ant
    cell_stat_prm = CellStatPrm(
        phyCellId=np.uint16(cell_id),
        nRxAnt=np.uint16(num_rx_ant),
        nTxAnt=np.uint16(num_tx_ant),
        nRxAntSrs=np.uint16(num_rx_ant_srs),
        nPrbUlBwp=np.uint16(num_prb_ul_bwp),
        nPrbDlBwp=np.uint16(num_prb_dl_bwp),
        mu=np.uint8(mu)
    )

    cuphy_tracker = CuPHYTracker(
        memoryFootprint=[]
    )

    pusch_stat_prms = PuschStatPrms(
        outInfo=[cuphy_tracker],
        WFreq=w_freq_array,
        WFreq4=w_freq4_array,
        WFreqSmall=w_freq_small_array,
        ShiftSeq=shift_seq_array,
        UnShiftSeq=unshift_seq_array,
        ShiftSeq4=shift_seq4_array,
        UnShiftSeq4=unshift_seq4_array,
        enableCfoCorrection=np.uint8(enable_cfo_correction),
        enableToEstimation=np.uint8(enable_to_estimation),
        enablePuschTdi=np.uint8(enable_pusch_tdi),
        enableDftSOfdm=np.uint8(0),  # Disable this feature now.
        enableTbSizeCheck=np.uint8(1),  # Always enabled.
        ldpcnIterations=np.uint8(10),  # To be deprecated.
        ldpcEarlyTermination=np.uint8(0),
        ldpcUseHalf=np.uint8(1),
        ldpcAlgoIndex=np.uint8(0),
        ldpcFlags=np.uint8(2),
        ldpcKernelLaunch=ldpc_kernel_launch,
        ldpcMaxNumItrAlgo=LdpcMaxItrAlgoType.LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT,
        fixedMaxNumLdpcItrs=np.uint8(10),
        nMaxLdpcHetConfigs=np.uint32(32),
        polarDcdrListSz=np.uint8(8),
        chEstAlgo=PuschChEstAlgoType(ch_est_algo),
        enablePerPrgChEst=np.uint8(enable_per_prg_chest),
        eqCoeffAlgo=eq_coeff_algo,
        enableUlRxBf=np.uint8(enable_ul_rx_bf),
        enableRssiMeasurement=np.uint8(0),
        enableSinrMeasurement=np.uint8(0),
        stream_priority=0,
        nMaxCells=np.uint16(1),
        nMaxCellsPerSlot=np.uint16(1),
        cellStatPrms=[cell_stat_prm],
        nMaxTbs=np.uint32(0),
        nMaxCbsPerTb=np.uint32(0),
        nMaxTotCbs=np.uint32(0),
        nMaxRx=np.uint32(0),
        nMaxPrb=np.uint32(273),
        enableEarlyHarq=np.uint8(0),
        enableDeviceGraphLaunch=np.uint8(0),
        earlyHarqProcNodePriority=np.int32(0),
        dbg=PuschStatDbgPrms(
            outFileName=debug_file_name,
            descrmOn=np.uint8(1),
            enableApiLogging=np.uint8(0)
        ),
        workCancelMode=PuschWorkCancelMode.PUSCH_NO_WORK_CANCEL  # update as needed
    )

    return pusch_stat_prms


def pusch_config_to_dyn_prms(
        cuda_stream: int,
        rx_data: List[np.ndarray],
        slot: int,
        pusch_configs: List[PuschConfig]) -> PuschDynPrms:
    """Convert pyAerial PuschConfig to format that the cuPHY PUSCH components want."""

    cell_dyn_prm = PuschCellDynPrm(
        cellPrmStatIdx=np.uint16(0),
        cellPrmDynIdx=np.uint16(0),
        slotNum=np.uint16(slot)
    )

    ue_grp_prms = []
    ue_prms = []
    num_ues = 0

    for ue_grp_idx, pusch_config in enumerate(pusch_configs):
        num_ues_in_grp = len(pusch_config.ue_configs)
        ue_grp_prms.append(PuschUeGrpPrm(
            cellPrmIdx=0,
            dmrsDynPrm=PuschDmrsPrm(
                dmrsAddlnPos=np.uint8(pusch_config.dmrs_add_ln_pos),
                dmrsMaxLen=np.uint8(pusch_config.dmrs_max_len),
                numDmrsCdmGrpsNoData=np.uint8(pusch_config.num_dmrs_cdm_grps_no_data),
                dmrsScrmId=np.uint16(pusch_config.dmrs_scrm_id)
            ),
            startPrb=np.uint16(pusch_config.start_prb),
            nPrb=np.uint16(pusch_config.num_prbs),
            prgSize=np.uint16(pusch_config.prg_size),
            nUplinkStreams=np.uint16(pusch_config.num_ul_streams),
            puschStartSym=np.uint8(pusch_config.start_sym),
            nPuschSym=np.uint8(pusch_config.num_symbols),
            dmrsSymLocBmsk=np.uint16(dmrs_bit_array_to_fapi(pusch_config.dmrs_syms)),
            rssiSymLocBmsk=np.uint16(dmrs_bit_array_to_fapi(pusch_config.dmrs_syms)),
            uePrmIdxs=list(np.array(range(num_ues, num_ues + num_ues_in_grp), dtype=np.uint16))
        ))

        for ue_params in pusch_config.ue_configs:
            ue_prms.append(PuschUePrm(
                pduBitmap=np.uint16(1),
                ueGrpIdx=np.uint16(ue_grp_idx),
                scid=np.uint8(ue_params.scid),
                dmrsPortBmsk=np.uint16(ue_params.dmrs_ports),
                mcsTable=np.uint8(ue_params.mcs_table),
                mcsIndex=np.uint8(ue_params.mcs_index),
                TBSize=np.uint32(ue_params.tb_size),
                targetCodeRate=np.uint16(ue_params.code_rate),
                qamModOrder=np.uint8(ue_params.mod_order),
                rv=np.uint8(ue_params.rv),
                rnti=np.uint16(ue_params.rnti),
                dataScramId=np.uint16(ue_params.data_scid),
                nUeLayers=np.uint8(ue_params.layers),
                ndi=np.uint8(ue_params.ndi),
                harqProcessId=np.uint8(ue_params.harq_process_id),
                # The following hard-coded.
                i_lbrm=np.uint8(0),
                maxLayers=np.uint8(4),
                maxQm=np.uint8(8),
                n_PRB_LBRM=np.uint16(273),
                enableTfPrcd=np.uint8(0),  # Disabled, not supported by pyAerial.
            ))

        num_ues += num_ues_in_grp

    cell_grp_dyn_prm = PuschCellGrpDynPrm(
        cellPrms=[cell_dyn_prm],
        ueGrpPrms=ue_grp_prms,
        uePrms=ue_prms,
    )

    data_in = PuschDataIn(
        tDataRx=[data.astype(np.complex64) for data in rx_data]
    )

    data_out = PuschDataOut(
        harqBufferSizeInBytes=np.zeros([num_ues], dtype=np.uint32),
        totNumTbs=np.zeros([1], dtype=np.uint32),
        totNumCbs=np.zeros([1], dtype=np.uint32),
        totNumPayloadBytes=np.zeros([1], dtype=np.uint32),
        totNumUciSegs=np.zeros([1], dtype=np.uint16),
        cbCrcs=np.ones([1000], dtype=np.uint32),
        tbCrcs=np.ones([num_ues], dtype=np.uint32),
        tbPayloads=np.zeros([200000], dtype=np.uint8),
        uciPayloads=None,
        uciCrcFlags=None,
        numCsi2Bits=None,
        startOffsetsCbCrc=np.zeros([num_ues], dtype=np.uint32),
        startOffsetsTbCrc=np.zeros([num_ues], dtype=np.uint32),
        startOffsetsTbPayload=np.zeros([num_ues], dtype=np.uint32),
        taEsts=np.zeros([num_ues], dtype=float),
        rssi=np.zeros([1], dtype=float),
        rsrp=np.zeros([num_ues], dtype=float),
        noiseVarPreEq=np.zeros([num_ues], dtype=float),
        noiseVarPostEq=np.zeros([num_ues], dtype=float),
        sinrPreEq=np.zeros([num_ues], dtype=float),
        sinrPostEq=np.zeros([num_ues], dtype=float),
        cfoHz=np.zeros([num_ues], dtype=float),
        HarqDetectionStatus=np.zeros([num_ues], dtype=np.uint8),
        CsiP1DetectionStatus=np.zeros([num_ues], dtype=np.uint8),
        CsiP2DetectionStatus=np.zeros([num_ues], dtype=np.uint8),
        preEarlyHarqWaitStatus=np.zeros([1], dtype=np.uint8),
        postEarlyHarqWaitStatus=np.zeros([1], dtype=np.uint8),
    )

    data_in_out = PuschDataInOut(
        harqBuffersInOut=[]
    )
    pusch_dyn_prms = PuschDynPrms(
        phase1Stream=cuda_stream,
        phase2Stream=cuda_stream,
        setupPhase=PuschSetupPhase.PUSCH_SETUP_PHASE_1,
        procModeBmsk=np.uint64(0),  # Controls PUSCH mode (e.g., will use CUDA graphs
                                    # if least significant bit is 1; streams if 0)
        waitTimeOutPreEarlyHarqUs=np.uint16(1000),
        waitTimeOutPostEarlyHarqUs=np.uint16(1500),
        cellGrpDynPrm=cell_grp_dyn_prm,
        dataIn=data_in,
        dataOut=data_out,
        dataInOut=data_in_out,
        cpuCopyOn=np.uint8(1),
        statusOut=PuschStatusOut(
            status=PuschStatusType.CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE,
            cellPrmStatIdx=np.uint16(0),
            ueIdx=np.uint16(0)
        ),
        dbg=PuschDynDbgPrms(
            enableApiLogging=np.uint8(0)
        ))

    return pusch_dyn_prms


def get_pusch_dyn_prms(  # pylint: disable=too-many-arguments, too-many-locals
        cuda_stream: int,
        rx_data: List[np.ndarray],
        num_ues: int,
        slot: int = 0,

        # UE group parameters.
        num_dmrs_cdm_grps_no_data: int = 2,
        dmrs_scrm_id: int = 41,
        start_prb: int = 0,
        num_prbs: int = 273,
        prg_size: int = 1,
        num_ul_streams: int = 1,
        dmrs_syms: Optional[List[int]] = None,
        dmrs_max_len: int = 2,
        dmrs_add_ln_pos: int = 1,
        start_sym: int = 2,
        num_symbols: int = 12,

        # UE parameters.
        scids: Optional[List[int]] = None,
        layers: Optional[List[int]] = None,
        dmrs_ports: Optional[List[int]] = None,
        rntis: Optional[List[int]] = None,
        data_scids: Optional[List[int]] = None,

        # CW parameters.
        mcs_tables: Optional[List[int]] = None,
        mcs_indices: Optional[List[int]] = None,
        target_code_rates: Optional[List[int]] = None,
        mod_orders: Optional[List[int]] = None,
        tb_sizes: Optional[List[int]] = None,
        rvs: Optional[List[int]] = None,
        ndis: Optional[List[int]] = None) -> PuschDynPrms:
    """Get a PuschDynPrms object based on given parameters.

    This gives a simple PuschDynPrms object for a single UE group, with default
    values for many parameters. It is intended for creating the PuschDynPrms
    for example when not all parameters are needed, such as when using only certain
    PUSCH receiver algorithms and not the full Rx chain, or when not all parameters
    are of interest, for example in some simulations.

    Args:
        cuda_stream (int): CUDA stream on which pipeline is launched.
        rx_data (List[np.ndarray]): List of tensors with each tensor (indexed by
            `cellPrmDynIdx`) representing the receive slot buffer of a cell in the cell group.
        num_ues (int): Number of UEs.
        slot (int): Slot number.
        num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
            [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
        dmrs_scrm_id (int): DMRS scrambling ID.
        start_prb (int): Start PRB index of the UE group allocation.
        num_prbs (int): Number of allocated PRBs for the UE group.
        prg_size (int): Size of PRG in PRB for the UE group.
        num_ul_streams (int): Number of allocated streams for the UE group.
        dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
            single-symbol DMRS or single- or double-symbol DMRS.
        dmrs_add_ln_pos (int): Number of additional DMRS positions.
        start_sym (int): Start OFDM symbol index for the UE group allocation.
        num_symbols (int): Number of symbols in the UE group allocation.
        scids (List[int]): DMRS sequence initialization for each UE
            [TS38.211, sec 7.4.1.1.2].
        layers (List[int]): Number of layers for each UE.
        dmrs_ports (List[int]): DMRS ports for each UE.
        rntis (List[int]): RNTI for each UE.
        data_scids (List[int]): Data scrambling IDs for each UE, more precisely
            `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
        mcs_tables (List[int]): MCS table to use for each UE (see TS 38.214).
        mcs_indices (List[int]): MCS indices for each UE.
        target_code_rates (List[int]): Code rate for each UE. This is the number of information
            bits per 1024 coded bits expressed in 0.1 bit units.
        mod_orders (List[int]): Modulation order for each UE.
        tb_sizes (List[int]): TB size in bytes for each UE.
        rvs (List[int]): Redundancy version per UE (default: 0 for each UE).
        ndis (List[int]): New data indicator per UE (default: 1 for each UE).

    Returns:
        PuschDynPrms: PUSCH dynamic parameters.
    """
    # Set the default values.
    if dmrs_syms is None:
        dmrs_syms = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    if scids is None:
        valid_scids = [0, 0, 1, 1, 0, 0, 1, 1]
        scids = valid_scids[:num_ues]
    if layers is None:
        layers = list(num_ues * [1, ])
    if dmrs_ports is None:
        valid_dmrs_ports = [1, 4, 2, 8, 16, 64, 32, 128]
        dmrs_ports = valid_dmrs_ports[:num_ues]
    if rntis is None:
        rntis = list(np.arange(1, num_ues + 1))
    if data_scids is None:
        data_scids = list(np.arange(1, num_ues + 1))

    if mcs_tables is None:
        mcs_tables = [0, ] * num_ues
    if mcs_indices is None:
        mcs_indices = list(num_ues * [0, ])
    if target_code_rates is None:
        target_code_rates = list(num_ues * [1930, ])
    if mod_orders is None:
        mod_orders = list(num_ues * [2, ])
    if tb_sizes is None:
        tb_sizes = [96321, ] * num_ues
    if rvs is None:
        rvs = [0, ] * num_ues
    if ndis is None:
        ndis = [1, ] * num_ues

    pusch_ue_configs = []
    for ue_idx in range(num_ues):
        pusch_ue_config = PuschUeConfig(
            scid=scids[ue_idx],
            layers=layers[ue_idx],
            dmrs_ports=dmrs_ports[ue_idx],
            rnti=rntis[ue_idx],
            data_scid=data_scids[ue_idx],
            mcs_table=mcs_tables[ue_idx],
            mcs_index=mcs_indices[ue_idx],
            code_rate=target_code_rates[ue_idx],
            mod_order=mod_orders[ue_idx],
            tb_size=tb_sizes[ue_idx],
            rv=rvs[ue_idx],
            ndi=ndis[ue_idx]
        )
        pusch_ue_configs.append(pusch_ue_config)

    pusch_configs = [PuschConfig(
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
        ue_configs=pusch_ue_configs
    )]

    pusch_dyn_prms = pusch_config_to_dyn_prms(
        cuda_stream=cuda_stream,
        rx_data=rx_data,
        slot=slot,
        pusch_configs=pusch_configs
    )

    return pusch_dyn_prms


def get_pusch_dyn_prms_phase_2(
        pusch_dyn_prms_phase1: PuschDynPrms,
        harq_buffer: Union[int, List[int]]) -> PuschDynPrms:
    """Get dynamic PUSCH phase 2 setup parameters."""
    if isinstance(harq_buffer, int):
        harq_buffer = [harq_buffer]

    pusch_dyn_prms_phase2 = pusch_dyn_prms_phase1._replace(
        setupPhase=PuschSetupPhase.PUSCH_SETUP_PHASE_2,
        dataInOut=PuschDataInOut(harqBuffersInOut=harq_buffer)
    )
    return pusch_dyn_prms_phase2
