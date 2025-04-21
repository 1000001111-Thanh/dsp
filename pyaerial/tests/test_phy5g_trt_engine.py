# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Tests for trt_engine.py."""
import pathlib
import pytest
import os

import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import engine_from_bytes
from polygraphy.backend.trt.runner import TrtRunner

from aerial.phy5g.algorithms import TrtEngine
from aerial.phy5g.algorithms import TrtTensorPrms

# Test cases.
model_configs = [
    {
        'name': 'LLRNet',
        'max_batch_size': 42588,
        'batch_size': 42588,
        'model_file': os.path.join(os.environ.get('HOME'), 'models/llrnet.trt'),
        'inputs': [TrtTensorPrms('input', (2,), np.float32)],
        'outputs': [TrtTensorPrms('dense_1', (8,), np.float32)]
    },
    {
        'name': 'LLRNet',
        'max_batch_size': 42588,
        'batch_size': 12345,
        'model_file': os.path.join(os.environ.get('HOME'), 'models/llrnet.trt'),
        'inputs': [TrtTensorPrms('input', (2,), np.float32)],
        'outputs': [TrtTensorPrms('dense_1', (8,), np.float32)]
    },
    {
        'name': 'NeuralRx',
        'model_file': os.path.join(os.environ.get('HOME'), 'models/neural_rx.trt'),
        'max_batch_size': 1,
        'batch_size': 1,
        'inputs': [TrtTensorPrms('rx_slot_real', (3276, 12, 4), np.float32),
                   TrtTensorPrms('rx_slot_imag', (3276, 12, 4), np.float32),
                   TrtTensorPrms('h_hat_real', (4914, 1, 4), np.float32),
                   TrtTensorPrms('h_hat_imag', (4914, 1, 4), np.float32),
                   TrtTensorPrms('active_dmrs_ports', (1,), np.float32),
                   TrtTensorPrms('dmrs_ofdm_pos', (3,), np.int32),
                   TrtTensorPrms('dmrs_subcarrier_pos', (6,), np.int32)],
        'outputs': [TrtTensorPrms('output_1', (2, 1, 3276, 12), np.float32),
                    TrtTensorPrms('output_2', (1, 3276, 12, 8), np.float32)]
    }
]


@pytest.mark.parametrize(
    "model_config",
    model_configs,
    ids=[config["name"] for config in model_configs]
)
def test_trt_engine(cuda_stream, model_config):
    """Test TrtEngine."""
    model_path = str(pathlib.Path(model_config["model_file"]).resolve())

    trt_engine = TrtEngine(
        trt_model_file=model_path,
        max_batch_size=model_config["max_batch_size"],
        input_tensors=model_config["inputs"],
        output_tensors=model_config["outputs"]
    )

    ref_engine = engine_from_bytes(bytes_from_path(model_path))

    # Run multiple times to check that all the buffers get set correctly.
    num_batches = 2
    for _ in range(num_batches):

        # Generate random input.
        feed_dict = {}
        for input_prms in model_config["inputs"]:
            shape = (model_config["batch_size"],) + input_prms.dims
            input_tensor = np.random.randn(*shape).astype(input_prms.data_type)
            feed_dict[input_prms.name] = input_tensor

        output_tensors = trt_engine.run(feed_dict)

        # Reference output using the polygraphy package.
        with TrtRunner(ref_engine) as ref_trt_runner:
            ref_output = ref_trt_runner.infer(feed_dict)

            for name, tensor in output_tensors.items():
                assert np.allclose(ref_output[name], tensor)
