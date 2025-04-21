# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - RSRP and pre-/post-equalizer SINR estimation."""
from typing import List
from typing import Tuple

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.params import get_pusch_stat_prms
from aerial.phy5g.params import PuschConfig
from aerial.phy5g.params import PuschUeConfig
from aerial.phy5g.params import pusch_config_to_dyn_prms


class RsrpEstimator:
    """RSRP, post- and pre-equalizer SINR estimator class.

    This class implements RSRP estimation as well as post- and pre-equalizer SINR
    estimation for PUSCH receiver pipeline.
    """
    def __init__(self,
                 num_rx_ant: int,
                 enable_pusch_tdi: int,
                 cuda_stream: int = None) -> None:
        """Initialize RsrpEstimator.

        Args:
            num_rx_ant (int): Number of receive antennas.
            enable_pusch_tdi (int): Whether time-interpolation is used in computing equalizer
                coefficients. This impacts post-equalizer SINR.
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        pusch_stat_prms = get_pusch_stat_prms(
            num_rx_ant=num_rx_ant,
            enable_pusch_tdi=enable_pusch_tdi
        )
        # pylint: disable=no-member
        self._pusch_params = pycuphy.PuschParams()
        self._pusch_params.set_stat_prms(pusch_stat_prms)

        self.rsrp_estimator = pycuphy.RsrpEstimator(self.cuda_stream)
        self.rsrp = None
        self.pre_eq_sinr = None
        self.post_eq_sinr = None
        self.noise_var_post_eq = None

    def estimate(self,
                 channel_est: List[np.ndarray],
                 ree_diag_inv: List[np.ndarray],
                 noise_var_pre_eq: np.ndarray,
                 pusch_configs: List[PuschConfig] = None,
                 num_ues: int = None,
                 num_prbs: int = None,
                 dmrs_add_ln_pos: int = None,
                 layers: List[int] = None) -> Tuple[List[np.ndarray],
                                                    List[np.ndarray],
                                                    List[np.ndarray]]:
        """Run RSRP and post- and pre-equalizer SINR estimation.

        Args:
            channel_est (List[np.ndarray]):  The channel estimates as a
                Rx ant x layer x frequency x time Numpy array, per UE group.
            ree_diag_inv  (List[np.ndarray]): Inverse of post-equalizer residual error covariance
                diagonal, per UE group.
            noise_var_pre_eq (np.ndarray): Average pre-equalizer noise variance in dB. One value
                per UE group.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given (only one UE group supported in that case).
            num_ues (int): Number of UEs in the UE group.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_add_ln_pos (int): Number of additional DMRS positions. This is used to derive the
                total number of DMRS symbols.
            layers (List[int]): Number of layers for each UE.

        Returns:
            List[np.ndarray], List[np.ndarray], List[np.ndarray]: A tuple containing:

            - *List[np.ndarray]*: RSRP values per UE.

            - *List[np.ndarray]*: Pre-equalization SINR values per UE.

            - *List[np.ndarray]*: Post-equalization SINR values per UE.

        """
        # If pusch_configs is given, use that directly. Else all the other parameters
        # need to be given (only a single UE group).
        if pusch_configs is None:

            # In this case all the other parameters need to be set.
            if num_ues is None:
                raise ValueError("Argument num_ues is not set!")
            if num_prbs is None:
                raise ValueError("Argument num_prbs is not set!")
            if dmrs_add_ln_pos is None:
                raise ValueError("Argument dmrs_add_ln_pos is not set!")
            if layers is None:
                raise ValueError("Argument layers is not set!")

            pusch_ue_configs = []
            for ue_idx in range(num_ues):
                pusch_ue_config = PuschUeConfig(
                    layers=layers[ue_idx],
                )
                pusch_ue_configs.append(pusch_ue_config)

            pusch_configs = [PuschConfig(
                num_prbs=num_prbs,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                ue_configs=pusch_ue_configs
            )]

        pusch_dyn_prms = pusch_config_to_dyn_prms(
            cuda_stream=self.cuda_stream,
            rx_data=[np.zeros((3276, 14, 1))],  # Dummy empty Rx data (Rx data not needed).
            slot=0,  # Not relevant here.
            pusch_configs=pusch_configs
        )
        self._pusch_params.set_dyn_prms(pusch_dyn_prms)

        self.rsrp = self.rsrp_estimator.estimate(
            channel_est,
            ree_diag_inv,
            np.float32(noise_var_pre_eq),
            self._pusch_params
        )

        self.pre_eq_sinr = self.rsrp_estimator.get_sinr_pre_eq()
        self.post_eq_sinr = self.rsrp_estimator.get_sinr_post_eq()
        self.noise_var_post_eq = self.rsrp_estimator.get_info_noise_var_post_eq()

        return self.rsrp, self.pre_eq_sinr, self.post_eq_sinr
