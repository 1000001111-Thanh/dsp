from .my_pusch_config import MyPUSCHConfig
from .my_encoder import MyTBEncoder
from .my_decoder import MyTBDecoder
from .my_pusch_pilot_pattern import MyPUSCHPilotPattern

from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.nr import PUSCHPilotPattern, LayerMapper, LayerDemapper, PUSCHLSChannelEstimator
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LinearDetector
from sionna.phy.nr.utils import generate_prng_seq
from sionna.phy.channel import AWGN, OFDMChannel, gen_single_sector_topology as gen_topology
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi, UMa, CDL
from sionna.phy import Block


import tensorflow as tf
import numpy as np


NUM_TX = NUM_RX = 1
NUM_STREAMS_PER_TX = 1


class MySimulator(Block):
    def __init__(self,
                 pusch_config:MyPUSCHConfig,
                 channel:OFDMChannel=None,
                 precision=None,
                 **kwargs):
        
        super().__init__(precision=precision, **kwargs)

        self.pusch_config = pusch_config

        self.bSrc = BinarySource()

        self.tbEnc = MyTBEncoder(target_tb_size=pusch_config.tb_size,
                            num_coded_bits=pusch_config.num_coded_bits,
                            target_coderate=pusch_config.tb.target_coderate,
                            num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol,
                            num_layers=pusch_config.num_layers,
                            n_rnti=pusch_config.n_rnti,
                            n_id=pusch_config.tb.n_id,
                            channel_type="PUSCH",
                            codeword_index=0,
                            use_scrambler=True,
                            verbose=False)
        
        
        self.cMapper = Mapper("qam", pusch_config.tb.num_bits_per_symbol)

        self.lMapper = LayerMapper(num_layers=pusch_config.num_layers)
    
        # Define the resource grid.
        resource_grid = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=pusch_config.num_subcarriers,
            subcarrier_spacing=pusch_config.carrier.subcarrier_spacing*1e3,
            num_tx=NUM_TX,
            num_streams_per_tx=NUM_STREAMS_PER_TX,
            cyclic_prefix_length=min(pusch_config.num_subcarriers, 288),
            num_guard_carriers=(0,0),
            dc_null=False,
            pilot_pattern=MyPUSCHPilotPattern([pusch_config])
        )

        self.rgMapper = ResourceGridMapper(resource_grid)        
        
        self.AWGN = AWGN()
        if channel:
            self.channel = channel
        else:
            Ue_Antenna = Antenna(polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=pusch_config._my_config.Carrier_frequency)

            Gnb_AntennaArray = AntennaArray(num_rows=pusch_config._my_config.Sys.NRxAnt//2,
                                    num_cols=1,
                                    polarization="dual",
                                    polarization_type="cross",
                                    antenna_pattern="38.901",
                                    carrier_frequency=pusch_config._my_config.Carrier_frequency)
            
            channel_model = CDL(model='C',
                                delay_spread=150*1e-9,
                                carrier_frequency=pusch_config._my_config.Carrier_frequency,
                                ut_array=Ue_Antenna,
                                bs_array=Gnb_AntennaArray,
                                direction="uplink",
                                min_speed=1,
                                max_speed=1)
            
            self.channel = OFDMChannel(channel_model=channel_model, resource_grid=resource_grid,
                                    add_awgn=False, normalize_channel=True, return_channel=True)
            

        self.chEst = PUSCHLSChannelEstimator(
                        resource_grid,
                        pusch_config.dmrs.length,
                        pusch_config.dmrs.additional_position,
                        pusch_config.dmrs.num_cdm_groups_without_data,
                        interpolation_type='nn')

        rxtx_association = np.ones([NUM_RX, NUM_TX], bool)
        stream_management = StreamManagement(rxtx_association, pusch_config.num_layers)
        self.det = LinearDetector("lmmse", "bit", "maxlog", resource_grid, stream_management,
                                    "qam", pusch_config.tb.num_bits_per_symbol)

        self.lDemapper = LayerDemapper(self.lMapper, num_bits_per_symbol=pusch_config.tb.num_bits_per_symbol)
        self.tbDec = MyTBDecoder(self.tbEnc)  
        
        mask = self.rgMapper._resource_grid.pilot_pattern._mask
        pilots = self.rgMapper._resource_grid.pilot_pattern._pilots

        self.ref = tf.tensor_scatter_nd_update(
                tf.zeros_like(mask, dtype=self.cdtype),
                tf.where(mask==1),
                tf.reshape(pilots, [-1])
            )
        
    def update_pilot_patterns(self, pilots, mask=None):
        """Channel Estimation and Detection will reflect this update since they reference the same object."""
        pilot_pattern = self.rgMapper._resource_grid.pilot_pattern
        pilot_pattern.pilots = pilots
        if mask: 
            pilot_pattern._mask = mask
        else: 
            mask = pilot_pattern._mask
        self.ref = tf.tensor_scatter_nd_update(
            tf.zeros_like(mask, dtype=self.cdtype),
            tf.where(mask==1),
            tf.reshape(pilots, [-1]))
        
    def update_scrambling_sequence(self, rnti=None, nid=None):
        assert not all(rnti, nid) == None, ""
        if rnti:
            self.pusch_config.n_rnti = rnti
        if nid:
            self.pusch_config.tb.n_id = nid

        self.tbEnc.scrambler.c_init = self.pusch_config._scb_c_init

    
    @property
    def channel(self):
        return self._channel
    
    @channel.setter
    def channel(self, channel):
        self._channel = channel

    @tf.function(jit_compile=True)
    def call(self, batch_size, no_scaling, gen_prng_seq=None, return_tx_iq=False, return_channel=False):
        if gen_prng_seq:
            b = tf.repeat(
                tf.reshape(
                    tf.constant(generate_prng_seq(NUM_TX * self.tbEnc.tb_size, gen_prng_seq), dtype=self.rdtype),
                    [1, NUM_TX, self.tbEnc.tb_size]
                ),
                repeats=batch_size,
                axis=0
            )

        else:
            b = self.bSrc([batch_size, NUM_TX, self.tbEnc.tb_size])

        c = self.tbEnc(b)
        x_map = self.cMapper(c)
        x_layer = self.lMapper(x_map)
        x = self.rgMapper(x_layer)

        y, h = self.channel(x)
        no = no_scaling * tf.math.reduce_variance(y, axis=[-1,-2,-3,-4])
        y = self.AWGN(y, no)

        if return_channel:
            if return_tx_iq:
                return b, c, y, x, h
            return b, c, y, h
        
        if return_tx_iq:
            return b, c, y, x
        return b, c, y
    

    def rec(self, y, snr_ = 1e3):
        no_ = tf.math.reduce_variance(y, axis=[-1,-2,-3,-4])/(snr_ + 1)
        h_hat, err_var = self.chEst(y, no_)
        print(h_hat.shape)
        llr_det = self.det(y, h_hat, err_var, no_)
        print(llr_det.shape)
        llr_layer = self.lDemapper(llr_det)
        print(llr_layer.shape)
        b_hat, tb_crc_status = self.tbDec(llr_layer)
        print(b_hat.shape)
        return h_hat, llr_det, b_hat, tb_crc_status

    def per(self, y, h, no):       
        h_hat, err_var = h, 0.
        llr_det = self.det(y, h_hat, err_var, no)
        llr_layer = self.lDemapper(llr_det)
        b_hat, tb_crc_status = self.tbDec(llr_layer)

        return h_hat, llr_det, b_hat, tb_crc_status