################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
################################################################################

"Models for PUSCH DMRS-based channel estimation"

import torch

import torch.nn as nn
import torch.nn.functional as F

from utils import sfft, isfft


class ResidualBlock(nn.Module):
    """
    Residual block used in Channel Estimator model.
    """
    def __init__(self, num_conv_channels, num_res, dilation):
        super(ResidualBlock, self).__init__()

        # Layer normalization over the last three dimensions: time, frequency, conv 'channels'
        self._norm1 = nn.LayerNorm((num_conv_channels, num_res))
        self._conv1 = nn.Conv1d(in_channels=num_conv_channels,
                                out_channels=num_conv_channels,
                                kernel_size=3,
                                padding=dilation,
                                dilation=dilation,
                                bias=False)
        self._norm2 = nn.LayerNorm((num_conv_channels, num_res))
        self._conv2 = nn.Conv1d(in_channels=num_conv_channels,
                                out_channels=num_conv_channels,
                                kernel_size=3,
                                padding=dilation,
                                dilation=dilation,
                                bias=False)

    def forward(self, inputs):
        z = self._conv1(inputs)
        z = self._norm1(z)
        z = F.relu(z)
        z = self._conv2(z)
        z = self._norm2(z)
        z = z + inputs  # skip connection
        z = F.relu(z)

        return z


class ChannelEstimator(nn.Module):
    """
    Implements the model that enhances LS channel estimation.
    """
    def __init__(self, num_res: int, num_conv_channels: int = 32, freq_interp_factor: int = 2):
        """
        Initializes channel estimator model.

        num_res: number of subcarriers with reference signals

        num_conv_channels: number of internal channels in convolutional layers
                           Impacts model complexity, not input/output size.

        freq_interp_factor: factor for interpolating in frequency.
            If 1, the size of the output will match input. If 2, output size will be
            twice the input size, by adding an interpolation layer at the end.
        """

        super(ChannelEstimator, self).__init__()
        self.freq_interp_factor = freq_interp_factor

        # Input convolution
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=num_conv_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.ReLU())

        # Residual blocks
        self.res_block_1 = ResidualBlock(num_conv_channels, num_res, dilation=1)
        self.res_block_2 = ResidualBlock(num_conv_channels, num_res, dilation=3)

        # Output convolution
        self.output_conv = nn.Conv1d(in_channels=num_conv_channels,
                                     out_channels=2,
                                     kernel_size=3,
                                     padding=1,
                                     bias=False)

        # Interpolation
        self.fcn = nn.Linear(in_features=2 * num_res, out_features=2 * num_res * freq_interp_factor)

    def forward(self, h):
        # Expected shape:            [batch size, num subcarriers]

        # Fourier transform with roll (to center information post-transform)
        h = sfft(h)             # [batch size, num subcarriers]

        z = torch.cat((h.real[..., None], h.imag[..., None]), axis=-1)
        # z: [batch size, num subcarriers, 2 (re & im)]

        z = z.permute(0, 2, 1)     # [batch size, 2, num_res]

        z = self.input_conv(z)     # [batch size, num_conv_channels, num_res]

        # Residual blocks
        z = self.res_block_1(z)    # [batch size, num_conv_channels, num_res]
        z = self.res_block_2(z)    # [batch size, num_conv_channels, num_res]

        z = self.output_conv(z)    # [batch size, 2, num_res]

        z = z.permute(0, 2, 1)     # [batch size, num_res, 2]

        if self.freq_interp_factor > 1:
            # FCN interpolation
            z = self.fcn(z.reshape((z.shape[0], -1)))  # [batch size, num_res*interp* 2]
            z = z.reshape((z.shape[0], -1, 2))         # [batch size, num_res*interp, 2]

        z = torch.view_as_complex(z.contiguous())      # [batch size, subcarriers]

        # Undo Fourier transform
        z = isfft(z)

        return z


class ComplexMSELoss(nn.Module):
    """
    Equivalent to torch.MSELoss() but for complex numbers.
    Note: torch.MSELoss([real, imag]) = 2* torch.MSELoss([complex]), due to normalization
    """
    def __init__(self):
        super(ComplexMSELoss, self).__init__()

    def forward(self, output: torch.complex64, target: torch.complex64):
        diff = output - target
        den = torch.linalg.norm((target * torch.conj(target)).mean())
        return torch.linalg.norm((diff * torch.conj(diff)).mean()) / den
