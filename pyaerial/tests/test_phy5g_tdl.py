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
import os


def tdl_chan_numpy(tdl_cfg, n_tti, cuda_stream):
    """Test TDL channel using numpy for Tx input signal, save data into H5 files."""
    # Step 1: create random input signals using numpy
    tx_signal_size = tdl_cfg.n_cell * tdl_cfg.n_ue * \
        tdl_cfg.time_signal_length_per_ant * tdl_cfg.n_tx_ant
    if tx_signal_size > 0:
        tx_signal_in = np.empty(tx_signal_size, dtype=np.complex64)
    else:
        tx_signal_in = None

    # Step 2: create TDL channel object
    tdl = pycuphy.TdlChan(
        tdl_cfg=tdl_cfg,
        tx_time_signal_in_cpu=tx_signal_in,
        rand_seed=0,
        stream_handle=cuda_stream
    )

    # Step 3: run test and save H5 file(s)
    for tti_idx in range(0, n_tti):
        # generate input signal
        if tx_signal_size > 0:
            tx_signal_in.real = np.random.rand(tx_signal_size)
            tx_signal_in.imag = np.random.rand(tx_signal_size)

        # run TDL test
        tdl.run(tti_idx * 5e-4)  # time stamp, assuming 500 us per slot

        # save H5 files in selected TTIs
        if tti_idx in [0, n_tti / 2, n_tti - 1]:
            pad_file_name_ending = f"_TTI{tti_idx}_numpy"
            tdl.save_tdl_chan_to_h5_file(pad_file_name_ending)

            out_filename = (
                f"tdlChan_{tdl_cfg.n_cell}cell{tdl_cfg.n_ue}Ue_"
                f"{tdl_cfg.n_tx_ant}x{tdl_cfg.n_rx_ant}_"
                f"{tdl_cfg.delay_profile}{int(tdl_cfg.delay_spread)}_"
                f"dopp{int(tdl_cfg.max_doppler_shift)}_"
                f"cfo{int(tdl_cfg.cfo_hz)}_"
                f"runMode{tdl_cfg.run_mode}_"
                f"freqConvert{tdl_cfg.freq_convert_type}_"
                f"scSampling{tdl_cfg.sc_sampling}_"
                f"FP32{pad_file_name_ending}.h5"
            )
            error_str = f"File '{out_filename}' does not exist in the current directory."
            assert os.path.isfile(out_filename), error_str
            os.remove(out_filename)


def tdl_chan_gpu_only(tdl_cfg, n_tti, cuda_stream):
    """Test TDL channel using GPU memory for tx input signal, save data into H5 files."""

    # Step 1: create buffer for random input signals and obtain GPU memory address
    tx_signal_size = tdl_cfg.n_cell * tdl_cfg.n_ue * \
        tdl_cfg.time_signal_length_per_ant * tdl_cfg.n_tx_ant
    if tx_signal_size > 0:
        tx_signal_in = np.empty(tx_signal_size, dtype=np.complex64)
        tx_signal_in_gpu = cuda.mem_alloc(tx_signal_in.nbytes)
    else:
        tx_signal_in = None
        tx_signal_in_gpu = 0

    # Step 2: create TDL channel object
    tdl = pycuphy.TdlChan(
        tdl_cfg=tdl_cfg,
        tx_time_signal_in_gpu=tx_signal_in_gpu,
        rand_seed=0,
        stream_handle=cuda_stream
    )

    # Step 3: run test and save H5 file(s)
    for tti_idx in range(0, n_tti):
        # generate input signal
        if tx_signal_size > 0:
            tx_signal_in.real = np.random.rand(tx_signal_size)
            tx_signal_in.imag = np.random.rand(tx_signal_size)
            cuda.memcpy_htod(tx_signal_in_gpu, tx_signal_in)

        # run TDL test
        tdl.run(tti_idx * 5e-4)  # time stamp, assuming 500 us per slot

        # save H5 files in selected TTIs
        if tti_idx in [0, n_tti / 2, n_tti - 1]:
            pad_file_name_ending = f"_TTI{tti_idx}_gpuOnly"
            tdl.save_tdl_chan_to_h5_file(pad_file_name_ending)

            out_filename = (
                f"tdlChan_{tdl_cfg.n_cell}cell{tdl_cfg.n_ue}Ue_"
                f"{tdl_cfg.n_tx_ant}x{tdl_cfg.n_rx_ant}_"
                f"{tdl_cfg.delay_profile}{int(tdl_cfg.delay_spread)}_"
                f"dopp{int(tdl_cfg.max_doppler_shift)}_"
                f"cfo{int(tdl_cfg.cfo_hz)}_"
                f"runMode{tdl_cfg.run_mode}_"
                f"freqConvert{tdl_cfg.freq_convert_type}_"
                f"scSampling{tdl_cfg.sc_sampling}_"
                f"FP32{pad_file_name_ending}.h5"
            )
            error_str = f"File '{out_filename}' does not exist in the current directory."
            assert os.path.isfile(out_filename), error_str
            os.remove(out_filename)


@pytest.mark.parametrize(
    "tdl_type, n_tti, run_mode, numpy_indicator", [
        ('A', 100, 0, 0),
        ('A', 100, 0, 1),
        ('A', 100, 1, 0),
        ('A', 100, 1, 1),
        ('C', 100, 0, 0),
        ('C', 100, 0, 1),
        ('C', 100, 1, 0),
        ('C', 100, 1, 1)
    ])
def test_tdl_chan(tdl_type, n_tti, run_mode, numpy_indicator, cuda_stream):
    """Main test function of TDL channel

    Paremeters:
    - tdl_type: The type of TDL channel to use for the test
        - 'A' - TDLA30-10, 'B' - TDLB100-400, 'C' - TDLC300-100
    - n_tti: number of TTIs in test
        - time stamp is tti_idx * 5e-4 assuming 500 us per time slot
        - tti_idx is (0, n_tti)
    - run_mode: The mode in which the test should be executed
        - 0: only time channel
        - 1: time + freq channel on prbg
        - 2: time + freq channel on prbg and sc
    - numpy_indicator: 1 - run test with numpy; 0 - run test with GPU momery directly
    - cuda_stream: cuda_stream to run test
    """
    try:
        # TDL configuration
        tdl_cfg = pycuphy.TdlConfig()
        tdl_cfg.delay_profile = tdl_type
        match tdl_type:
            case 'A':
                tdl_cfg.delay_spread = 30
                tdl_cfg.max_doppler_shift = 10
            case 'B':
                tdl_cfg.delay_spread = 100
                tdl_cfg.max_doppler_shift = 400
            case 'C':
                tdl_cfg.delay_spread = 300
                tdl_cfg.max_doppler_shift = 100
            case _:
                raise NotImplementedError("Unsupported TDL channel type")
        tdl_cfg.run_mode = run_mode

        if numpy_indicator:
            tdl_chan_numpy(tdl_cfg, n_tti, cuda_stream)
        else:
            tdl_chan_gpu_only(tdl_cfg, n_tti, cuda_stream)

    except Exception as e:
        assert False, f"Error running TDL channel test: {e}"
