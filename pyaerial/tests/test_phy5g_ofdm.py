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
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda


def compare_arrays(array1, array2, gpu_indicator, size):
    """
    Compare two GPU or CPU arrays to check if they are close.

    Parameters:
    - array1: first array
    - array2: second array
    - gpu_indicator:
        - 0: array1 and array2 are numpy arraies
        - 1: array1 and array2 are GPU arries, i.e., pycuda._driver.DeviceAllocation
    - size: int, number of elements to compare

    Returns:
    - no returns, assert failure if arrays are not equal.
    """
    error_str = "OFDM modulation and demodulation results do not match, test FAIL!"
    if gpu_indicator:
        # allocate host memory
        host_array1 = np.empty(size, dtype=np.complex64)
        host_array2 = np.empty(size, dtype=np.complex64)

        # copy data from GPU to host
        cuda.memcpy_dtoh(host_array1, array1)
        cuda.memcpy_dtoh(host_array2, array2)

        assert np.allclose(host_array1, host_array2, rtol=1e-5, atol=1e-5), error_str
    else:
        assert np.allclose(array1, array2, rtol=1e-5, atol=1e-5), error_str


def ofdm_mod_demod_numpy(cuphy_carrier_prms, n_tti, cuda_stream):
    """
        Test OfdmModulate and OfdmDeModulate using numpy for frequency data samples.
    """

    # Step 1: allocate numpy buffers
    freq_data_size = cuphy_carrier_prms.n_tx_layer * cuphy_carrier_prms.n_symbol_slot \
        * cuphy_carrier_prms.n_sc
    # since we only do OFDM mod + demod, input and output data have the same size
    freq_in = np.empty(freq_data_size, dtype=np.complex64)
    freq_out = np.empty(freq_data_size, dtype=np.complex64)

    # Step 2: create OFDM modulation and demodulation objects
    ofdm_mod = pycuphy.OfdmModulate(cuphy_carrier_prms=cuphy_carrier_prms,
                                    freq_data_in_cpu=freq_in,
                                    stream_handle=cuda_stream
                                    )
    time_data_gpu = ofdm_mod.get_time_data_out()
    ofdm_demod = pycuphy.OfdmDeModulate(cuphy_carrier_prms=cuphy_carrier_prms,
                                        time_data_in_gpu=time_data_gpu,
                                        freq_data_out_cpu=freq_out,
                                        prach=0,
                                        stream_handle=cuda_stream
                                        )
    # Step 3: run test and compare results
    for tti_idx in range(0, n_tti):
        # generate freq in data using numpy
        freq_in.real = np.random.rand(freq_data_size)
        freq_in.imag = np.random.rand(freq_data_size)
        # output frequency data will be stored in GPU memory by ofdmDeModulate

        # run OFDM modulation and demodulation
        ofdm_mod.run()  # or ofdm_mod.run(freq_in_new) with a numpy array (to be created)
        ofdm_demod.run()  # or ofdm_demod.run(freq_out_new) with a numpy array (to be created)
        compare_arrays(
            array1=freq_in,
            array2=freq_out,
            gpu_indicator=0,
            size=freq_data_size
        )


def ofdm_mod_demod_gpu_only(cuphy_carrier_prms, n_tti, cuda_stream):
    """
        Test OfdmModulate and OfdmDeModulate using GPU memory address for frequency data samples.
    """

    # Step 1: allocate GPU buffers
    freq_data_size = cuphy_carrier_prms.n_tx_layer * cuphy_carrier_prms.n_symbol_slot \
        * cuphy_carrier_prms.n_sc
    freq_in = np.empty(freq_data_size, dtype=np.complex64)
    # since we only do OFDM mod + demod, input and output data have the same size
    freq_in_gpu = cuda.mem_alloc(freq_in.nbytes)
    freq_out_gpu = cuda.mem_alloc(freq_in.nbytes)

    # Step 2: create OFDM modulation and demodulation objects
    ofdm_mod = pycuphy.OfdmModulate(cuphy_carrier_prms=cuphy_carrier_prms,
                                    freq_data_in_gpu=freq_in_gpu,
                                    stream_handle=cuda_stream
                                    )
    time_data_in_gpu = ofdm_mod.get_time_data_out()
    ofdm_demod = pycuphy.OfdmDeModulate(cuphy_carrier_prms=cuphy_carrier_prms,
                                        time_data_in_gpu=time_data_in_gpu,
                                        freq_data_out_gpu=freq_out_gpu,
                                        prach=0,
                                        stream_handle=cuda_stream
                                        )

    # Step 3: run test and compare results
    for tti_idx in range(0, n_tti):
        # generate freq in data and copy to GPU
        freq_in.real = np.random.rand(freq_data_size)
        freq_in.imag = np.random.rand(freq_data_size)
        cuda.memcpy_htod(freq_in_gpu, freq_in)
        # output frequency data will be stored in GPU memory by ofdmDeModulate

        # run OFDM modulation and demodulation
        ofdm_mod.run()
        ofdm_demod.run()
        compare_arrays(
            array1=freq_in_gpu,
            array2=freq_out_gpu,
            gpu_indicator=1,
            size=freq_data_size
        )


@pytest.mark.parametrize(
    "n_sc, n_tti, numpy_indicator", [
        (1632, 100, 0),
        (1632, 100, 1),
        (3276, 100, 0),
        (3276, 100, 1)
    ]
)
def test_ofdm_mod_demod(n_sc, n_tti, numpy_indicator, cuda_stream):
    """
    Main test function of OfdmModulate and OfdmDeModulate.

    Paremeters:
    - n_sc: number of sub-carriers
    - n_tti: number of TTIs in test
    - numpy_indicator: 1 - run test with numpy; 0 - run test with GPU momery directly
    - cuda_stream: cuda_stream to run test
    """
    try:
        # carrier configurations
        cuphy_carrier_prms = pycuphy.CuphyCarrierPrms()
        cuphy_carrier_prms.n_sc = n_sc

        if numpy_indicator:
            ofdm_mod_demod_numpy(cuphy_carrier_prms, n_tti, cuda_stream)
        else:
            ofdm_mod_demod_gpu_only(cuphy_carrier_prms, n_tti, cuda_stream)

    except Exception as e:
        assert False, f"Error running OFDM test: {e}"
