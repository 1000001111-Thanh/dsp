# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - Carrier frequency offset and timing advance estimation (for equalization)."""
from typing import List
from typing import Tuple

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.params import get_pusch_stat_prms
from aerial.phy5g.params import pusch_config_to_dyn_prms
from aerial.phy5g.params import PuschConfig
from aerial.phy5g.params import PuschUeConfig


class CfoTaEstimator:
    """CFO and TA estimator class.

    This class implements an algorithm for carrier frequency offset and timing advance
    estimation. It calls the corresponding cuPHY algorithms and provides the estimates
    as needed for other cuPHY algorithms.

    It needs channel estimates as its input.
    """
    def __init__(self,
                 num_rx_ant: int,
                 mu: int = 1,
                 enable_cfo_correction: bool = True,
                 enable_to_estimation: bool = True,
                 cuda_stream: int = None) -> None:
        """Initialize CfoTaEstimator.

        Args:
            num_rx_ant (int): Number of receive antennas.
            mu (int): Numerology. Values in [0, 3]. Default: 1.
            enable_cfo_correction (int): Enable/disable CFO correction:

                - 0: Disable.
                - 1: Enable (default).

            enable_to_estimation (int): Enable/disable time offset estimation:

                - 0: Disable.
                - 1: Enable (default).

            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        pusch_stat_prms = get_pusch_stat_prms(
            num_rx_ant=num_rx_ant,
            mu=mu,
            enable_cfo_correction=int(enable_cfo_correction),
            enable_to_estimation=int(enable_to_estimation)
        )
        # pylint: disable=no-member
        self._pusch_params = pycuphy.PuschParams()
        self._pusch_params.set_stat_prms(pusch_stat_prms)

        self.cfo_ta_estimator = pycuphy.CfoTaEstimator(cuda_stream)
        self.cfo_est = None
        self.cfo_hz = None
        self.ta = None

    def estimate(self,
                 channel_est: List[np.ndarray],
                 pusch_configs: List[PuschConfig] = None,
                 num_ues: int = None,
                 num_prbs: int = None,
                 dmrs_syms: List[int] = None,
                 dmrs_max_len: int = None,
                 dmrs_add_ln_pos: int = None,
                 layers: List[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate carrier frequency offset and timing advance.

        Args:
            channel_est (List[np.ndarray]): The channel estimates as a
                Rx ant x layer x frequency x time Numpy array, per UE group.
                Note: Currently this only supports a single UE group, i.e. the length of the list
                is one.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given (only one UE group supported in that case).
            num_ues (int): Number of UEs in the UE group.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol. The length of the list equals
                the number of symbols in the slot. 0 means no DMRS in the symbol and 1 means
                the symbol is a DMRS symbol.
            dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
                single-symbol DMRS or single- or double-symbol DMRS.
            dmrs_add_ln_pos (int): Number of additional DMRS positions.
            layers (List[int]): Number of layers for each UE. The length of the list equals the
                number of UEs.

        Returns:
            np.ndarray, np.ndarray: A tuple containing:

            - *np.ndarray*: Carrier frequency offset per UE, in Hz.

            - *np.ndarray*: Timing offset per UE, in microseconds.
        """
        # If pusch_configs is given, use that directly. Else all the other parameters
        # need to be given (only a single UE group).
        if pusch_configs is None:

            # In this case all the other parameters need to be set.
            if num_ues is None:
                raise ValueError("Argument num_ues is not set!")
            if num_prbs is None:
                raise ValueError("Argument num_prbs is not set!")
            if dmrs_syms is None:
                raise ValueError("Argument dmrs_syms is not set!")
            if dmrs_max_len is None:
                raise ValueError("Argument dmrs_max_len is not set!")
            if dmrs_add_ln_pos is None:
                raise ValueError("Argument dmrs_add_ln_pos is not set!")
            if layers is None:
                raise ValueError("Argument layers is not set!")

            pusch_ue_configs = [PuschUeConfig(layers=layers[ue]) for ue in range(num_ues)]
            pusch_configs = [PuschConfig(
                num_prbs=num_prbs,
                dmrs_syms=dmrs_syms,
                dmrs_max_len=dmrs_max_len,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                ue_configs=pusch_ue_configs
            )]

        pusch_dyn_prms = pusch_config_to_dyn_prms(
            cuda_stream=self.cuda_stream,
            rx_data=[np.zeros((3276, 14, 1))],  # Dummy empty Rx data (Rx data not needed).
            slot=0,  # Not used.
            pusch_configs=pusch_configs
        )

        self._pusch_params.set_dyn_prms(pusch_dyn_prms)

        self.cfo_est = self.cfo_ta_estimator.estimate(channel_est, self._pusch_params)

        self.cfo_hz = self.cfo_ta_estimator.get_cfo_hz()
        self.ta = self.cfo_ta_estimator.get_ta()

        return self.cfo_hz, self.ta
