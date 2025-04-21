# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - fading channel."""

# pylint: disable=no-member
import numpy as np
import pycuda.driver as drv  # type: ignore
import pycuda.autoinit  # type: ignore  # noqa: F401  # pylint: disable=unused-import
from aerial import pycuphy


class FadingChan:
    """Fading channel class.

    This class implements the fading channel that processes the frequency
    Tx samples and outputs frequency Rx samples. It includes OFDM modulation,
    tapped delay line (TDL) channel, OFDM demodulation, and adds noise based on
    input SNR.
    """
    def __init__(self,
                 cuphy_carrier_prms: pycuphy.CuphyCarrierPrms,  # type: ignore
                 tdl_type: str,
                 freq_in: np.ndarray,
                 rand_seed: int) -> None:
        """Initialize the FadingChan class.

        Args:
            cuphy_carrier_prms (pycuphy.CuphyCarrierPrms): Carrier parameters for the channel.
            tdl_type (str): Type of the channel model (e.g., 'A', 'B', 'C').
            freq_in (np.ndarray): Input frequency Tx samples.
            rand_seed (int): Random seed for TDL channel generation.
        """
        # Step 1: carrier and TDL configurations.
        self.stream = drv.Stream()
        self.prach = 0  # TODO: Add support for PRACH
        # TDL configurations, default is TDLA30-5-Low
        self.tdl_cfg = pycuphy.TdlConfig()
        self.tdl_cfg.delay_profile = tdl_type
        match tdl_type:
            case 'A':
                self.tdl_cfg.delay_spread = 30
                self.tdl_cfg.max_doppler_shift = 10
            case 'B':
                self.tdl_cfg.delay_spread = 100
                self.tdl_cfg.max_doppler_shift = 400
            case 'C':
                self.tdl_cfg.delay_spread = 300
                self.tdl_cfg.max_doppler_shift = 100
            case _:
                raise NotImplementedError("Unsupported TDL channel type")
        self.tdl_cfg.run_mode = 0  # Only TDL time channel and filter Tx signal

        # Check input buffer data size match with config.
        self.freq_data_in_size = cuphy_carrier_prms.n_tx_layer * cuphy_carrier_prms.n_symbol_slot \
            * cuphy_carrier_prms.n_sc
        assert self.freq_data_in_size == freq_in.size
        self.freq_in = freq_in  # Save the numpy buffer location.

        # Allocate output buffer data size match with config.
        self.freq_data_out_size = cuphy_carrier_prms.n_tx_layer * cuphy_carrier_prms.n_symbol_slot \
            * cuphy_carrier_prms.n_sc
        self.freq_out_nois_free = np.empty(self.freq_data_out_size, dtype=np.complex64)

        # Step 2: Create OFDM modulation, TDL channel, OFDM demodulation
        # Create OFDM modulation.
        self.ofdm_mod = pycuphy.OfdmModulate(
            cuphy_carrier_prms=cuphy_carrier_prms,
            freq_data_in_cpu=self.freq_in,
            stream_handle=self.stream.handle
        )

        # Create TDL channel.
        self.tdl_cfg.time_signal_length_per_ant = int(
            self.ofdm_mod.get_time_data_length() / cuphy_carrier_prms.n_tx_layer
        )  # Input data length per antenna.
        self.tx_time_signal_in_gpu = self.ofdm_mod.get_time_data_out()
        self.tdl = pycuphy.TdlChan(
            tdl_cfg=self.tdl_cfg,
            tx_time_signal_in_gpu=self.tx_time_signal_in_gpu,
            rand_seed=rand_seed,
            stream_handle=self.stream.handle
        )

        # CreateOFDM demodulation.
        rx_time_signal_out_gpu = self.tdl.get_rx_time_signal_out()
        self.ofdm_demod = pycuphy.OfdmDeModulate(
            cuphy_carrier_prms=cuphy_carrier_prms,
            time_data_in_gpu=rx_time_signal_out_gpu,
            freq_data_out_cpu=self.freq_out_nois_free,
            prach=self.prach,
            stream_handle=self.stream.handle
        )

    def add_noise_with_snr_complex(self, snr_db: float) -> np.ndarray:
        """Add Gaussian noise to a complex signal with a specified SNR.

        Args:
            snr_db (float): Desired Signal-to-Noise Ratio in decibels.

        Returns:
            np.ndarray: The frequency-domain signal with noise added.
        """
        # TODO: Change this to GPU processing for lower latency.
        # Convert SNR from dB to linear scale.
        snr_linear = 10 ** (snr_db / 10)
        # Generate Gaussian noise for both real and imaginary parts.
        noise_real = np.sqrt(0.5 / snr_linear) * np.random.randn(*self.freq_out_nois_free.shape)
        noise_imag = np.sqrt(0.5 / snr_linear) * np.random.randn(*self.freq_out_nois_free.shape)
        noise = noise_real + 1j * noise_imag

        # Add noise to the signal.
        return self.freq_out_nois_free + noise

    def run(self, tti_idx: int, snr_db: float, freq_in: np.ndarray = None) -> np.ndarray:
        """Run the fading channel.

        Args:
            tti_idx (int): TTI index.
            snr_db (float): Signal-to-Noise Ratio in dB.
            freq_in (np.ndarray): Frequency domain input samples.

        Returns:
            np.ndarray: Frequency domain samples after channel processing.
        """
        if freq_in is not None:  # using new array as input
            self.ofdm_mod.run(freq_in)
        else:
            self.ofdm_mod.run()
        self.tdl.run(tti_idx * 5e-4)  # time stamp, assuming 500 us per slot
        self.ofdm_demod.run()  # Stream synchronize inside ofdm_demod.run().
        return self.add_noise_with_snr_complex(snr_db)  # Add noise in frequency domain on CPU.
