# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - SRS channel estimation."""
import os
from typing import List
from typing import NamedTuple
from typing import Tuple

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g import chest_filters


class SrsCellPrms(NamedTuple):
    """SRS cell parameters.

    A list of SRS cell parameters is given to the SRS channel estimator as input,
    one entry per cell.

    Args:
        slot_num (np.uint16): Slot number.
        frame_num (np.uint16): Frame number.
        srs_start_sym (np.uint8): SRS start symbol.
        num_srs_sym (np.uint8): Number of SRS symbols.
        num_rx_ant_srs (np.uint16): Number of SRS Rx antennas.
        mu (np.uint8): Subcarrier spacing parameter, see TS 38.211.
    """

    slot_num: np.uint16
    frame_num: np.uint16
    srs_start_sym: np.uint8
    num_srs_sym: np.uint8
    num_rx_ant_srs: np.uint16
    mu: np.uint8


class UeSrsPrms(NamedTuple):
    """UE SRS parameters.

    A list of UE SRS parameters is given to the SRS channel estimator as input,
    one entry per UE.

    Args:

        cell_idx (np.uint16): Index of cell user belongs to.
        num_ant_ports (np.uint8): Number of SRS antenna ports. 1,2, or 4.
        num_syms (np.uint8): Number of SRS symbols. 1,2, or 4.
        num_repetitions (np.uint8): Number of repititions. 1,2, or 4.
        comb_size (np.uint8): SRS comb size. 2 or 4.
        start_sym (np.uint8): Starting SRS symbol. 0 - 13.
        sequence_id (np.uint16): SRS sequence ID. 0 - 1023.
        config_idx (np.uint8): SRS bandwidth configuration idndex. 0 - 63.
        bandwidth_idx (np.uint8): SRS bandwidth index. 0 - 3.
        comb_offset (np.uint8): SRS comb offset. 0 - 3.
        cyclic_shift (np.uint8): Cyclic shift. 0 - 11.
        frequency_position (np.uint8): Frequency domain position. 0 - 67.
        frequency_shift (np.uint16): Frequency domain shift. 0 - 268.
        frequency_hopping (np.uint8): Freuqnecy hopping options. 0 - 3.
        resource_type (np.uint8): Type of SRS allocation. 0:
            Aperiodic. 1: Semi-persistent. 2: Periodic.
        periodicity (np.uint16): SRS periodicity in slots.
            0, 2, 3, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, 2560.
        offset (np.uint16): Slot offset value. 0 - 2569.
        group_or_sequence_hopping (np.uint8): Hopping configuration.
            0: No hopping. 1: Group hopping. 2: Sequence hopping.
        ch_est_buff_idx (np.uint16): Index of which buffer to store SRS estimates into.
        srs_ant_port_to_ue_ant_map (np.ndarray): Mapping between SRS antenna ports and UE antennas
            in channel estimation buffer: Store estimates for SRS antenna port i in
            srs_ant_port_to_ue_ant_map[i].
        prg_size (np.uint8): Number of PRBs per PRG.
    """

    cell_idx: np.uint16
    num_ant_ports: np.uint8
    num_syms: np.uint8
    num_repetitions: np.uint8
    comb_size: np.uint8
    start_sym: np.uint8
    sequence_id: np.uint16
    config_idx: np.uint8
    bandwidth_idx: np.uint8
    comb_offset: np.uint8
    cyclic_shift: np.uint8
    frequency_position: np.uint8
    frequency_shift: np.uint16
    frequency_hopping: np.uint8
    resource_type: np.uint8
    periodicity: np.uint16
    offset: np.uint16
    group_or_sequence_hopping: np.uint8
    ch_est_buff_idx: np.uint16
    srs_ant_port_to_ue_ant_map: np.ndarray
    prg_size: np.uint8


class SrsReport(NamedTuple):
    """SRS output report.

    This report is returned by the SRS channel estimator.

    Args:
        to_est_micro_sec (np.float32): Time offset estimate in microseconds.
        wideband_snr (np.float3): Wideband SNR.
        wideband_noise_energy (np.float32): Wideband noise energy.
        wideband_signal_energy (np.float32): Wideband signal energy.
        wideband_sc_corr (np.complex64): Wideband subcarrier correlation.
        wideband_cs_corr_ratio_db (np.float32):
        wideband_cs_corr_use (np.float32):
        wideband_cs_corr_not_use (np.float32):
    """
    to_est_micro_sec: np.float32
    wideband_snr: np.float32
    wideband_noise_energy: np.float32
    wideband_signal_energy: np.float32
    wideband_sc_corr: np.complex64
    wideband_cs_corr_ratio_db: np.float32
    wideband_cs_corr_use: np.float32
    wideband_cs_corr_not_use: np.float32


class SrsChannelEstimator:
    """SrsChannelEstimator class.

    This class implements SRS channel sounding for 5G NR.
    """
    def __init__(self, chest_params: dict = None) -> None:
        """Initialize SrsChannelEstimator.

        Args:
            chest_params (dict): Dictionary of channel estimation filters and parameters.
                Set to None to use defaults.
        """
        # Create the CUDA stream.
        self.stream = check_cuda_errors(cudart.cudaStreamCreate())

        if chest_params is None:
            # Default SRS channel estimation parameters are loaded from this file.
            filename = os.path.join(
                os.path.dirname(__file__),
                'chest_coeffs',
                'cuPhyChEstCoeffs.h5'
            )
            chest_params = chest_filters.srs_chest_params_from_hdf5(filename)

        # pylint: disable=no-member
        self.channel_estimator = pycuphy.SrsChannelEstimator(chest_params, self.stream)

    def estimate(self,
                 rx_data: np.ndarray,
                 num_srs_ues: int,
                 num_srs_cells: int,
                 num_prb_grps: int,
                 start_prb_grp: int,
                 srs_cell_prms: List[SrsCellPrms],
                 srs_ue_prms: List[UeSrsPrms]) -> Tuple[list, np.ndarray, list]:
        """Run SRS channel estimation.

        Args:
            rx_data (np.ndarray): Input RX data, size num_subcarriers x num_srs_sym x num_rx_ant.
            num_srs_ues (int): Number of UEs.
            num_srs_cells (int): Number of SRS cells.
            num_prb_grps (int): Number of PRB groups.
            start_prb_grp (int): Start PRB group.
            srs_cell_prms (List[SrsCellPrms]): List of SRS cell parameters, one per cell.
            srs_ue_prms (List[UeSrsPrms]): List of UE SRS parameters, one per UE.

        Returns:
            List[np.ndarray], np.ndarray, List[SrsReport]: A tuple containing:

            - *List[np.ndarray]*:
              A list of channel estimates, one per UE. The channel estimate is a
              num_prb_grps x num_rx_ant x num_tx_ant numpy array.
            - *np.ndarray*:
              SNRs per RB per UE.
            - *List[SrsReport]*:
              A list of SRS wideband statistics reports, one per UE.
        """
        ch_est = self.channel_estimator.estimate(
            np.complex64(rx_data),
            np.uint16(num_srs_ues),
            np.uint16(num_srs_cells),
            np.uint16(num_prb_grps),
            np.uint16(start_prb_grp),
            srs_cell_prms,
            srs_ue_prms
        )
        rb_snr_buffer = self.channel_estimator.get_rb_snr_buffer()
        srs_report = self.channel_estimator.get_srs_report()

        return ch_est, rb_snr_buffer, srs_report
