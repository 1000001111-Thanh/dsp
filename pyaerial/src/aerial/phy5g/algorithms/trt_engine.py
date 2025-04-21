# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - TensorRT engine."""
from dataclasses import dataclass
from typing import List

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.types import DataType


@dataclass
class TrtTensorPrms:
    """Class to hold the TRT input and output tensor parameters."""

    # Name of the tensor as in the TRT file.
    name: str

    # Tensor dimensions without batch dimension.
    dims: List[int]

    # Data type. Supported: np.float32, np.int32
    data_type: type = np.float32

    @property
    def cuphy_data_type(self) -> DataType:
        """Convert data type to cuPHY data type format."""
        if self.data_type == np.float32:
            cuphy_data_type = DataType.CUPHY_R_32F
        elif self.data_type == np.int32:
            cuphy_data_type = DataType.CUPHY_R_32I
        else:
            raise ValueError("Invalid data type (supported: np.float32, np.int32)")

        return cuphy_data_type


class TrtEngine:
    """TensorRT engine class.

    This class implements a simple wrapper around NVIDIA's TensorRT and its
    cuPHY API. It takes a TRT engine file as its input, along with the names
    and dimensions of the input and output tensors. The TRT engine file
    can be generated offline from an `.onnx` file using the `trtexec` tool.
    """

    def __init__(self,
                 trt_model_file: str,
                 max_batch_size: int,
                 input_tensors: List[TrtTensorPrms],
                 output_tensors: List[TrtTensorPrms],
                 cuda_stream: int = None) -> None:
        """Initialize TrtEngine.

        Args:
            trt_model_file (str): This is TRT engine (model) file.
            max_batch_size (int): Maximum batch size.
            input_tensors (List[TrtTensorPrms]): A mapping from tensor names to input tensor
                dimensions. The names are strings that must match with those found in the TRT model
                file, and the shapes are iterables of integers. The batch dimension is skipped.
            output_tensors (List[TrtTensorPrms]): A mapping from tensor names to output tensor
                dimensions. The names are strings that must match with those found in the TRT model
                file, and the shapes are iterables of integers. The batch dimension is skipped.
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        self.input_names = [tensor.name for tensor in input_tensors]
        self.input_dims = [tensor.dims for tensor in input_tensors]
        self.input_cuphy_data_types = [tensor.cuphy_data_type for tensor in input_tensors]
        self.input_data_types = [tensor.data_type for tensor in input_tensors]

        self.output_names = [tensor.name for tensor in output_tensors]
        self.output_dims = [tensor.dims for tensor in output_tensors]
        self.output_cuphy_data_types = [tensor.cuphy_data_type for tensor in output_tensors]
        self.output_data_types = [tensor.data_type for tensor in output_tensors]

        self.trt_engine = pycuphy.TrtEngine(  # pylint: disable=no-member
            trt_model_file,
            max_batch_size,
            self.input_names,
            self.input_dims,
            self.input_cuphy_data_types,
            self.output_names,
            self.output_dims,
            self.output_cuphy_data_types,
            self.cuda_stream
        )

    def run(self, input_tensors: dict) -> dict:
        """Run the TensorRT model.

        This runs the model using NVIDIA TensorRT engine.

        Args:
            input_tensors (dict): A mapping from input tensor names to the actual
                input tensors. The tensor names must match with those given in the initialization,
                and with those found in the TRT model. Actual batch size is read from
                the tensor size.

        Returns:
            dict: A mapping from output tensor names to the actual output tensors.
        """
        # Order the input tensors into a list.
        trt_input = []
        for index, name in enumerate(self.input_names):
            try:
                input_tensor = input_tensors[name]

                # Verify shape.
                input_dims = self.input_dims[index]
                if input_dims != input_tensor.shape[1:]:
                    raise ValueError(f"Tensor {name} has invalid shape!")

                if input_tensor.dtype != self.input_data_types[index]:
                    print(
                        f"Warning! Tensor {name} is not of the configured data type, casting..."
                    )
                    input_tensor = input_tensor.astype(self.input_data_types[index])

                trt_input.append(input_tensor)
            except KeyError:
                print(f"Tensor {name} not found in the inputs!")
                raise

        trt_output = self.trt_engine.run(trt_input)

        # Add the outputs into a dict.
        output_tensors = dict()
        for index, (output, output_name) in enumerate(zip(trt_output, self.output_names)):
            output_tensors[output_name] = output.astype(self.output_data_types[index])

        return output_tensors
