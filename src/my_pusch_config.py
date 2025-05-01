#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH configuration for the nr (5G) sub-package of the Sionna library.
"""
# pylint: disable=line-too-long

import numpy as np
from sionna.phy.nr.utils import generate_prng_seq
from sionna.phy.nr import PUSCHConfig, CarrierConfig, PUSCHDMRSConfig, TBConfig
from .data import MyConfig

class MyPUSCHConfig(PUSCHConfig):
    """
    The PUSCHConfig objects sets parameters for a physical uplink shared
    channel (PUSCH), as described in Sections 6.3 and 6.4 [3GPP38211]_.

    All configurable properties can be provided as keyword arguments during the
    initialization or changed later.

    Parameters
    ----------
    carrier_config : :class:`~sionna.nr.CarrierConfig` or `None`
        An instance of :class:`~sionna.nr.CarrierConfig`. If `None`, a
        :class:`~sionna.nr.CarrierConfig` instance with default settings
        will be created.

    pusch_dmrs_config : :class:`~sionna.nr.PUSCHDMRSConfig` or `None`
        An instance of :class:`~sionna.nr.PUSCHDMRSConfig`. If `None`, a
        :class:`~sionna.nr.PUSCHDMRSConfig` instance with default settings
        will be created.

    Example
    -------
    >>> pusch_config = PUSCHConfig(mapping_type="B")
    >>> pusch_config.dmrs.config_type = 2
    >>> pusch_config.carrier.subcarrier_spacing = 30
    """
    def __init__(self,
                 carrier_config:CarrierConfig=None,
                 pusch_dmrs_config:PUSCHDMRSConfig=None,
                 tb_config:TBConfig=None,
                 my_config:MyConfig=None,
                 slot_number:np.uint8=4,
                 frame_number:np.uint16=0,
                 **kwargs):
        super(PUSCHConfig, self).__init__(**kwargs)
        self._name = "PUSCH Configuration"
        self._my_config = my_config
        if my_config:
            self.carrier=CarrierConfig(
                n_cell_id=my_config.Sys.NCellId,
                cyclic_prefix="normal" if ~my_config.Sys.CpType else "extended",
                subcarrier_spacing=15*(2**my_config.Sys.Numerology),
                n_size_grid=my_config.Sys.BwpNRb,
                n_start_grid=my_config.Sys.BwpRbOffset,
                slot_number=slot_number,
                frame_number=frame_number
            )
            self.dmrs=PUSCHDMRSConfig(
                config_type=my_config.Ue.DmrsConfigurationType + 1,
                length=my_config.Ue.DmrsDuration,
                additional_position=my_config.Ue.DmrsAdditionalPosition,
                dmrs_port_set=my_config.Ue.DmrsPortSetIdx,
                n_id=[my_config.Ue.NnScIdId,my_config.Ue.NnScIdId] if my_config.Ue.NnScIdId else [my_config.Sys.NCellId, my_config.Sys.NCellId],
                n_scid=my_config.Ue.nScId,
                num_cdm_groups_without_data=my_config.Ue.NumDmrsCdmGroupsWithoutData,
                type_a_position=my_config.Ue.DmrsTypeAPosition
            )
            self.tb=TBConfig(
                channel_type='PUSCH',
                n_id=my_config.Ue.nId if my_config.Ue.nId else my_config.Sys.NCellId,
                mcs_table=my_config.Ue.McsTable + 1,
                mcs_index=my_config.Ue.Mcs
            )
            self.mapping_type='A' if ~my_config.Ue.PuschMappingType else 'B'
            self.n_size_bwp=my_config.Sys.BwpNRb
            self.n_start_bwp=my_config.Sys.BwpRbOffset
            self.num_layers=my_config.Ue.NLayers
            self.num_antenna_ports=len(my_config.Ue.DmrsPortSetIdx)
            self.precoding='non-codebook' if ~my_config.Ue.CodeBookBased else 'codebook'
            self.tpmi=my_config.Ue.Tpmi
            self.transform_precoding=False if ~my_config.Ue.TransformPrecoding else True
            self.n_rnti=my_config.Ue.Rnti
            self.symbol_allocation=[my_config.Ue.FirstSymb,my_config.Ue.NPuschSymbAll]
        else:
            self.carrier = carrier_config
            self.dmrs = pusch_dmrs_config
            self.tb = tb_config
        self.check_config()

    #-----------------------------#
    #---Configurable parameters---#
    #-----------------------------#

    @property
    def phy_cell_id(self):
        return self._carrier._n_cell_id
    
    @phy_cell_id.setter
    def phy_cell_id(self, value):
        self.carrier._n_cell_id = value
        self.tb._n_id = value
        self.dmrs._n_id = [value, value]

    @property
    def first_resource_block(self):
        """
        :class:`~sionna.phy.nr.CarrierConfig` : Carrier configuration
        """
        return self._my_config.Ue.FirstPrb

    @property
    def first_subcarrier(self):
        """
        :class:`~sionna.phy.nr.CarrierConfig` : Carrier configuration
        """
        return 12*self.first_resource_block

    @property
    def num_resource_blocks(self):
        """
        int, read-only : Number of allocated resource blocks for the
            PUSCH transmissions.
        """
        return self._my_config.Ue.NPrb

    @property
    def dmrs_grid(self):
        # pylint: disable=line-too-long
        """
        complex, [num_dmrs_ports, num_subcarriers, num_symbols_per_slot], read-only : Empty
            resource grid for each DMRS port, filled with DMRS signals

            This property returns for each configured DMRS port an empty
            resource grid filled with DMRS signals as defined in
            Section 6.4.1.1 [3GPP38211]. Not all possible options are implemented,
            e.g., frequency hopping and transform precoding are not available.

            This property provides the *unprecoded* DMRS for each configured DMRS port.
            Precoding might be applied to map the DMRS to the antenna ports. However,
            in this case, the number of DMRS ports cannot be larger than the number of
            layers.
        """
        # Check configuration
        self.check_config()

        # Configure DMRS ports set if it has not been set
        reset_dmrs_port_set = False
        if len(self.dmrs.dmrs_port_set)==0:
            self.dmrs.dmrs_port_set = list(range(self.num_layers))
            reset_dmrs_port_set = True

        # Generate empty resource grid for each port
        a_tilde = np.zeros([len(self.dmrs.dmrs_port_set),
                            self.num_subcarriers,
                            self.carrier.num_symbols_per_slot],
                            dtype=complex)
        first_subcarrier = self.first_subcarrier
        num_subcarriers = self.num_subcarriers

        # For every l_bar
        for l_bar in self.l_bar:

            # For every l_prime
            for l_prime in self.l_prime:

                # Compute c_init
                l = l_bar + l_prime
                c_init = self.c_init(l)
                # Generate RNG
                c = generate_prng_seq(first_subcarrier + num_subcarriers, c_init=c_init)
                if self.dmrs.config_type==1:
                    c = c[2*first_subcarrier//2:]
                else:
                    c = c[2*first_subcarrier//3:]

                # Map to QAM
                r = 1/np.sqrt(2)*((1-2*c[::2]) + 1j*(1-2*c[1::2]))

                # For every port in the dmrs port set
                for j_ind, _ in enumerate(self.dmrs.dmrs_port_set):

                    # For every n
                    for n in self.n:

                        # For every k_prime
                        for k_prime in [0, 1]:

                            if self.dmrs.config_type==1:
                                k = 4*n + 2*k_prime + \
                                    self.dmrs.deltas[j_ind]
                            else: # config_type == 2
                                k = 6*n + k_prime + \
                                    self.dmrs.deltas[j_ind]

                            a_tilde[j_ind, k, self.l_ref+l] = \
                                r[2*n + k_prime] * \
                                self.dmrs.w_f[k_prime][j_ind] * \
                                self.dmrs.w_t[l_prime][j_ind]

        # Amplitude scaling
        a = self.dmrs.beta*a_tilde

        # Reset DMRS port set if it was not set
        if reset_dmrs_port_set:
            self.dmrs.dmrs_port_set = []
        return a
    
    @property
    def _pilot_sequence(self):
        dmrs_grid = self.dmrs_grid
        return dmrs_grid[np.where(np.broadcast_to(self.dmrs_mask, dmrs_grid.shape))]
    
    @property
    def _scb_c_init(self):
        return self.n_rnti * 2**15 + self.tb.n_id

        