# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""pyAerial library - CSI-RS transmitter."""
from typing import List

from cuda import cudart  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.phy5g.types import CsiRsCellDynPrms
from aerial.phy5g.types import CsiRsPmwOneLayer


class CsiRsTx:
    """CSI-RS transmitter.

    This class implements CSI-RS transmission within a slot.
    """
    def __init__(self, num_prb_dl_bwp: List[int], cuda_stream: int = None) -> None:
        """Initialize CsiRsTx.

        Args:
            num_prb_dl_bwp (List[int]): Number of PRBs in DL BWP.
            cuda_stream (int): The CUDA stream. If not given, one will be created.
        """
        if cuda_stream is None:
            cuda_stream = check_cuda_errors(cudart.cudaStreamCreate())
        self.cuda_stream = cuda_stream

        self.csi_rs_tx = pycuphy.CsiRsTx(num_prb_dl_bwp)  # pylint: disable=no-member

    def run(self,
            csirs_cell_dyn_prms: List[CsiRsCellDynPrms],
            tx_buffers: List[np.ndarray],
            precoding_matrices: List[CsiRsPmwOneLayer] = None) -> List[np.ndarray]:
        """Run CSI-RS transmission.

        Args:
            csirs_cell_dyn_prms (List[CsiRsCellDynPrms]): A list of CSI-RS cell dynamic parameters,
                one entry per cell. See `CsiRsCellDynPrms`.
            tx_buffers (List[np.ndarray]): A list of transmit slot buffers, one per cell. These
                represent the slot buffers prior to inserting the CSI-RS.
            precoding_matrices (List[CsiRsPmwOneLayer]): A list of precoding matrices. This list
                gets indexed by the `pmw_prm_idx` field in `CsiRsRrcDynPrms` (part of
                `CsiRsCellDynPrms`).

        Returns:
            List[np.ndarray]: Transmit buffers for the slot for each cell after inserting CSI-RS.
        """
        precoding_matrices = precoding_matrices or []
        tx_buffers = self.csi_rs_tx.run(csirs_cell_dyn_prms,
                                        precoding_matrices,
                                        tx_buffers,
                                        self.cuda_stream)
        return tx_buffers
