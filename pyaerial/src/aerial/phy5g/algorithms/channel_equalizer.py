# Copyright (c) 2023-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - Channel equalization and soft demapping."""
from typing import List
from typing import Tuple

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.types import PuschEqCoefAlgoType
from aerial.phy5g.params import PuschConfig
from aerial.phy5g.params import PuschUeConfig
from aerial.phy5g.params import get_pusch_stat_prms
from aerial.phy5g.params import pusch_config_to_dyn_prms


class ChannelEqualizer:
    """Channel equalizer class.

    This class implements MMSE-based channel equalization along with soft
    demapping to get log-likelihood ratios for channel decoding.

    It needs channel estimates and noise and interference estimates as its input,
    along with the received data symbols.
    """
    def __init__(self,
                 num_rx_ant: int,
                 eq_coeff_algo: int,
                 enable_pusch_tdi: int,
                 cuda_stream: int = None) -> None:
        """Initialize ChannelEqualizer.

        Args:
            num_rx_ant (int): Number of receive antennas.
            eq_coeff_algo (int): Algorithm used to compute equalizer coefficients.

                - 0: Zero-forcing equalizer.
                - 1: MMSE with noise variance only.
                - 2: MMSE-IRC.
            enable_pusch_tdi (int): Whether to use time-domain interpolation.
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        pusch_stat_prms = get_pusch_stat_prms(
            num_rx_ant=num_rx_ant,
            enable_pusch_tdi=enable_pusch_tdi,
            eq_coeff_algo=PuschEqCoefAlgoType(eq_coeff_algo)
        )
        # pylint: disable=no-member
        self._pusch_params = pycuphy.PuschParams()
        self._pusch_params.set_stat_prms(pusch_stat_prms)

        self.channel_equalizer = pycuphy.ChannelEqualizer(self.cuda_stream)
        self.eq_sym = None
        self.llrs = None
        self.eq_coef = []  # type: List[np.ndarray]
        self.ree_diag_inv = []  # type: List[np.ndarray]

    def equalize(self,  # pylint: disable=too-many-arguments
                 *,
                 rx_slot: np.ndarray,
                 channel_est: List[np.ndarray],
                 lw_inv: List[np.ndarray],
                 noise_var_pre_eq: np.ndarray,
                 pusch_configs: List[PuschConfig] = None,
                 num_ues: int = None,
                 num_dmrs_cdm_grps_no_data: int = None,
                 start_prb: int = None,
                 num_prbs: int = None,
                 dmrs_syms: List[int] = None,
                 dmrs_max_len: int = None,
                 dmrs_add_ln_pos: int = None,
                 start_sym: int = None,
                 num_symbols: int = None,
                 layers: List[int] = None,
                 mod_orders: List[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Run equalization and soft demapping.

        This runs the cuPHY equalization for all UE groups included in `pusch_configs`.
        If this argument is not given, all the other arguments need to be given and cuPHY
        equalization is run only for a single UE group sharing the same time-frequency resources,
        i.e. having the same PRB allocation, and the same start symbol and number of allocated
        symbols.

        Args:
            rx_slot (np.ndarray): Input received data as a frequency x time x Rx antenna Numpy
                array with type `np.complex64` entries.
            channel_est (List[np.ndarray]):  The channel estimates as a
                Rx ant x layer x frequency x time Numpy array, per UE group.
            lw_inv  (List[np.ndarray]): Inverse of the Cholesky decomposition of the
                noise/interference covariance matrix per PRB, per UE group. The size of each entry
                in this list is number of Rx antennas x number of Rx antennas x number of PRBs.
            noise_var_pre_eq (np.ndarray): Average pre-equalizer noise variance in dB. One value
                per UE group.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given (only one UE group supported in that case).
            num_ues (int): Number of UEs in the UE group.
            num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
                [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
            start_prb (int): Start PRB index of the UE allocation.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol. The length of the list equals
                the number of symbols in the slot. 0 means no DMRS in the symbol and 1 means
                the symbol is a DMRS symbol.
            dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
                single-symbol DMRS or single- or double-symbol DMRS.
            dmrs_add_ln_pos (int): Number of additional DMRS positions.
            start_sym (int): Start symbol index for the UE group allocation.
            num_symbols (int): Number of symbols in the UE group allocation.
            layers (List[int]): Number of layers for each UE.
            mod_orders (List[int]): QAM modulation order for each UE.

        Returns:
            List[np.ndarray], List[np.ndarray]: A tuple containing:

            - *List[np.ndarray]*:
              Log-likelihood ratios for the received bits to be fed into
              decoding (rate matching). One Numpy array per UE group and the size of each
              Numpy array is 8 x number of layers x number of subcarriers x number of data
              symbols. The size of the first dimension is fixed to eight as modulations up
              to 256QAM are supported and cuPHY returns the same size independently of
              modulation. Only the first entries corresponding to the actual number of bits
              are used.
            - *List[np.ndarray]*:
              Equalized symbols, one Numpy array per UE group. The size of each
              Numpy array is equal to number of layers x number of subcarriers x number of data
              symbols.
        """
        # If pusch_configs is given, use that directly. Else all the other parameters
        # need to be given (only a single UE group).
        if pusch_configs is None:

            # In this case all the other parameters need to be set.
            if num_ues is None:
                raise ValueError("Argument num_ues is not set!")
            if num_dmrs_cdm_grps_no_data is None:
                raise ValueError("Argument num_dmrs_cdm_grps_no_data is not set!")
            if start_prb is None:
                raise ValueError("Argument start_prb is not set!")
            if num_prbs is None:
                raise ValueError("Argument num_prbs is not set!")
            if dmrs_syms is None:
                raise ValueError("Argument dmrs_syms is not set!")
            if dmrs_max_len is None:
                raise ValueError("Argument dmrs_max_len is not set!")
            if dmrs_add_ln_pos is None:
                raise ValueError("Argument dmrs_add_ln_pos is not set!")
            if start_sym is None:
                raise ValueError("Argument start_sym is not set!")
            if num_symbols is None:
                raise ValueError("Argument num_symbols is not set!")
            if layers is None:
                raise ValueError("Argument layers is not set!")
            if mod_orders is None:
                raise ValueError("Argument mod_orders is not set!")

            pusch_ue_configs = []
            for ue_idx in range(num_ues):
                pusch_ue_config = PuschUeConfig(
                    layers=layers[ue_idx],
                    mod_order=mod_orders[ue_idx]
                )
                pusch_ue_configs.append(pusch_ue_config)

            pusch_configs = [PuschConfig(
                num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
                start_prb=start_prb,
                num_prbs=num_prbs,
                dmrs_syms=dmrs_syms,
                dmrs_max_len=dmrs_max_len,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                start_sym=start_sym,
                num_symbols=num_symbols,
                ue_configs=pusch_ue_configs
            )]

        pusch_dyn_prms = pusch_config_to_dyn_prms(
            cuda_stream=self.cuda_stream,
            rx_data=[rx_slot],
            slot=0,  # Not relevant here.
            pusch_configs=pusch_configs
        )
        self._pusch_params.set_dyn_prms(pusch_dyn_prms)

        # Compute noise variance inverse from the pre-eq value (dB).
        # Note: This is trying to match what is done in cuPHY. A 0.5dB offset is used
        # to compensate for averaging.
        inv_noise_var_lin = np.power(10, - (noise_var_pre_eq - 0.5) / 10.)

        self.llrs = self.channel_equalizer.equalize(
            channel_est,
            lw_inv,
            np.float32(noise_var_pre_eq),
            np.float32(inv_noise_var_lin),
            self._pusch_params
        )

        self.eq_sym = self.channel_equalizer.get_data_eq()

        self.eq_coef = []
        for eq_coef in self.channel_equalizer.get_eq_coef():
            eq_coef = np.transpose(eq_coef, (0, 2, 4, 3, 1))
            eq_coef = np.reshape(eq_coef, (*eq_coef.shape[:3], -1))
            eq_coef = eq_coef.transpose(0, 1, 3, 2)
            self.eq_coef.append(eq_coef)

        self.ree_diag_inv = self.channel_equalizer.get_ree_diag_inv()

        return self.llrs, self.eq_sym
