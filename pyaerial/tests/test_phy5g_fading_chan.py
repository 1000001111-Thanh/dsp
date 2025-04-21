# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from aerial import pycuphy
import numpy as np
import pytest
from aerial.phy5g.chan_models import FadingChan


@pytest.mark.parametrize(
    "n_sc, tdl_type, n_tti", [
        (1632, 'A', 100),
        (1632, 'C', 100),
        (3276, 'A', 100),
        (3276, 'C', 100)
    ]
)
def test_fading_chan(n_sc, tdl_type, n_tti, snr_db=10):
    """
    Test the fading channel model with specified parameters.

    - n_sc: number of subcarriers
    - tdl_type:tDL channel model type (e.g., 'A', 'B', 'C')
    - n_tti: number of TTIs in test
    - snr_db: SNR in dB
    """
    try:
        # carriar parameters
        cuphy_carrier_prms = pycuphy.CuphyCarrierPrms()
        # allocate numpy buffers for freq in and freq out samples
        freq_data_in_size = cuphy_carrier_prms.n_tx_layer * cuphy_carrier_prms.n_symbol_slot \
            * cuphy_carrier_prms.n_sc
        freq_in = np.empty(freq_data_in_size, dtype=np.complex64)

        # creat FadingChan object
        fading_chan = FadingChan(
            cuphy_carrier_prms=cuphy_carrier_prms,
            tdl_type=tdl_type,
            freq_in=freq_in,
            rand_seed=0
        )

        # run FadingChan object with specific TTI index and SNR (dB)
        for tti_idx in range(0, n_tti):
            # generate freq in data using numpy
            freq_in.real = np.random.rand(freq_data_in_size)
            freq_in.imag = np.random.rand(freq_data_in_size)
            # run fading channel
            freq_out = fading_chan.run(
                tti_idx=tti_idx,
                snr_db=snr_db,
                # or freq_in=freq_in_new with a numpy array (to be created)
            )
            assert freq_out.size > 0, "freq_out is empty"

    except Exception as e:
        assert False, f"Error running fading channel test: {e}"
