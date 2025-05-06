from .my_pusch_config import MyPUSCHConfig
from .my_encoder import MyTBEncoder
from .my_decoder import MyTBDecoder
from .my_pusch_pilot_pattern import MyPUSCHPilotPattern
from .my_timer import tic, toc

from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.nr import LayerMapper, LayerDemapper, PUSCHLSChannelEstimator
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, LinearDetector
from sionna.phy.nr.utils import generate_prng_seq
from sionna.phy.channel import AWGN, OFDMChannel, gen_single_sector_topology as gen_topology
from sionna.phy.mimo import StreamManagement
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi
# from sionna.phy.channel import RayleighBlockFading
from sionna.phy import Block

import tensorflow as tf
import numpy as np
import copy

NUM_TX = NUM_RX = 1
NUM_STREAMS_PER_TX = 1


class MySimulator(Block):
    def __init__(self,
                 pusch_config:MyPUSCHConfig,
                 channel=None,
                 snr=1e9,
                 precision=None,
                 **kwargs):
        
        super().__init__(precision=precision, **kwargs)

        self.pusch_config = pusch_config.clone()
        self._tb_size = pusch_config.tb_size
        self._num_coded_bits = pusch_config.num_coded_bits
        self._target_coderate = pusch_config.tb.target_coderate
        self._num_bits_per_symbol = pusch_config.tb.num_bits_per_symbol
        self._num_layers=pusch_config.num_layers

        self.bSrc = BinarySource()

        self.tbEnc = MyTBEncoder(target_tb_size=self._tb_size,
                            num_coded_bits=self._num_coded_bits,
                            target_coderate=self._target_coderate,
                            num_bits_per_symbol=self._num_bits_per_symbol,
                            num_layers=self._num_layers,
                            n_rnti=pusch_config.n_rnti,
                            n_id=pusch_config.tb.n_id,
                            channel_type="PUSCH",
                            codeword_index=0,
                            use_scrambler=True,
                            verbose=False)
        
        self._cb_size = self.tbEnc._cb_size
        
        self.cMapper = Mapper("qam", self._num_bits_per_symbol)

        self.lMapper = LayerMapper(num_layers=self._num_layers)
    
        # Define the resource grid.
        self._pilot_pattern=MyPUSCHPilotPattern([pusch_config])

        num_subcarriers = pusch_config.num_subcarriers
        resource_grid = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=num_subcarriers,
            subcarrier_spacing=pusch_config.carrier.subcarrier_spacing*1e3,
            num_tx=NUM_TX,
            num_streams_per_tx=NUM_STREAMS_PER_TX,
            cyclic_prefix_length=min(num_subcarriers, 288),
            num_guard_carriers=(0,0),
            dc_null=False,
            pilot_pattern=self._pilot_pattern
        )
        
        self.rgMapper = ResourceGridMapper(resource_grid)

        self._pilot_ind = self.rgMapper._pilot_ind
        self._data_ind = self.rgMapper._data_ind
        self._rg_shape = self.rgMapper._rg_type.shape

        self.awgn = AWGN()
        self._snr = tf.Variable(snr, self.rdtype)

        if channel:
            self.channel = channel
        else:
            carrier_frequency = pusch_config._my_config.Carrier_frequency
            Ue_Antenna = Antenna(polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=carrier_frequency)

            Gnb_AntennaArray = AntennaArray(num_rows=pusch_config._my_config.Sys.NRxAnt//2,
                                    num_cols=1,
                                    polarization="dual",
                                    polarization_type="cross",
                                    antenna_pattern="38.901",
                                    carrier_frequency=carrier_frequency)
            channel_model = UMi(carrier_frequency = carrier_frequency,
                                        o2i_model = "high",
                                        ut_array = Ue_Antenna,
                                        bs_array = Gnb_AntennaArray,
                                        direction = "uplink",
                                        enable_pathloss = True,
                                        enable_shadow_fading = True)
            
            channel_model.set_topology(*gen_topology(1, 1, "umi"))

            self.channel = OFDMChannel(channel_model=channel_model, resource_grid=resource_grid,
                                    add_awgn=False, normalize_channel=True, return_channel=True)
        


        self.chEst = PUSCHLSChannelEstimator(
                        resource_grid,
                        pusch_config.dmrs.length,
                        pusch_config.dmrs.additional_position,
                        pusch_config.dmrs.num_cdm_groups_without_data,
                        interpolation_type='nn')

        rxtx_association = np.ones([NUM_RX, NUM_TX], bool)
        stream_management = StreamManagement(rxtx_association, self._num_layers)
        self.det = LinearDetector("lmmse", "bit", "maxlog", resource_grid, stream_management,
                                    "qam", self._num_bits_per_symbol)

        self.lDemapper = LayerDemapper(self.lMapper, num_bits_per_symbol=self._num_bits_per_symbol)
        self.tbDec = MyTBDecoder(self.tbEnc)  
    
    @property
    def reference(self):
        return tf.scatter_nd(self._pilot_ind,
                             tf.reshape(self._pilot_pattern.pilots, [-1]),
                             self._rg_shape)
    def get_r_rg(self, batch_size):
        r = self.reference
        return tf.broadcast_to(r, (batch_size, *r.shape))
    def get_c_rg(self, c):
        c_shape = c.shape
        c_sp = tf.reshape(c, (*c_shape[:-1], -1, self._num_layers, self._num_bits_per_symbol))
        c_sp = tf.transpose(c_sp,[1,3,2,4,0])
        c_rg = tf.scatter_nd(self._data_ind,
                    tf.reshape(c_sp, (-1, *c_sp.shape[3:])),
                    (*self._rg_shape,self._num_bits_per_symbol,c_shape[0]))
        return tf.transpose(c_rg,perm=[5,0,1,2,3,4])

    def update(self, *, rnti=None, slot_num=None, pci=None, rb_start=None, verbose=False):
        # Check that at least one parameter is not None
        # assert not all([rnti, slot_num, pci, rb_start]) == None, "At least one parameter must be provided"
        tic()
        pusch_config = self.pusch_config
        toc("assign pusch config")

        tic()
        cur_rnti = pusch_config.n_rnti
        cur_slot_num = pusch_config.carrier.slot_number
        cur_pci = pusch_config.phy_cell_id
        cur_rb_start = pusch_config.first_resource_block
        toc("store current pusch config")
        # # Update PUSCH configuration based on provided parameters

        tic()
        if rnti is not None:
            pusch_config.n_rnti = rnti
        if slot_num is not None:
            pusch_config.carrier.slot_number = slot_num
        if pci is not None:
            pusch_config.phy_cell_id = pci
        if rb_start is not None:
            pusch_config.first_resource_block = rb_start
        toc("update new pusch config")
        
        if verbose == True:
            tf.print("\nUpdating parameters:")
            tf.print(f" - RNTI: {cur_rnti} -> {pusch_config.n_rnti}")
            tf.print(f" - Slot number: {cur_slot_num} -> {pusch_config.carrier.slot_number}")
            tf.print(f" - PCI: {cur_pci} -> {pusch_config.phy_cell_id}")
            tf.print(f" - RB start: {cur_rb_start} -> {pusch_config.first_resource_block}")
        
        if rnti or pci:
            tic()
            c_init = pusch_config._scb_c_init
            toc("calculate c_init")
            tic()
            self.tbEnc.scrambler.c_init = c_init
            toc("update c_init")
        if slot_num or rb_start or pci:
            tic()
            seq = pusch_config._pilot_sequence
            toc("calculate pilot seq")
            tic()
            self._pilot_pattern.pilots = seq
            toc("update pilot_sequence")

    @property
    def channel(self):
        return self._channel
    
    @channel.setter
    def channel(self, channel):
        self._channel = channel
    
    def set_topology(self,
                     batch_size,
                     num_ut=1,
                     scenario="umi",
                     min_bs_ut_dist=None,
                     isd=None,
                     bs_height=None,
                     min_ut_height=None,
                     max_ut_height=None,
                     indoor_probability=None,
                     min_ut_velocity=None,
                     max_ut_velocity=None
                    ):
        self.channel._cir_sampler.set_topology(*gen_topology(batch_size,
                                                             num_ut, scenario,
                                                             min_bs_ut_dist, isd, bs_height,
                                                             min_ut_height, max_ut_height,
                                                             indoor_probability,
                                                             min_ut_velocity, max_ut_velocity))

    @property
    def snr(self):
        return self._snr
    
    @snr.setter
    def snr(self, snr):
        snr = tf.convert_to_tensor(snr, dtype=self.rdtype)
        self._snr.assign(snr)

    @tf.function(jit_compile=True)
    def call(self,
             batch_size,
             prng_seed=None,
             return_items=('b', 'c', 'y'),  # Thay thế các cờ bằng tham số điều khiển
             use_coded_bits=False):
        seq_len = self._tb_size if not use_coded_bits else self._cb_size
        if prng_seed:
            prng_seq = tf.constant(generate_prng_seq(NUM_TX * seq_len, prng_seed), dtype=self.rdtype)
            prng = tf.reshape(prng_seq, [1, NUM_TX, seq_len])
            prng = tf.repeat(prng, repeats=batch_size, axis=0)

            if use_coded_bits:
                b = None
                c = prng
            else:
                b = prng
                c = self.tbEnc(b)
        else:
            if use_coded_bits:
                b = None
                c = self.bSrc([batch_size, NUM_TX, seq_len])
            else:
                b = self.bSrc([batch_size, NUM_TX, seq_len])
                c = self.tbEnc(b)

        x_map = self.cMapper(c)
        x_layer = self.lMapper(x_map)
        x = self.rgMapper(x_layer)

        y, h = self.channel(x)
        no = tf.math.reduce_variance(y, axis=[-1,-2,-3,-4]) / self.snr
        y = self.awgn(y, no)

        # Tạo output map chứa tất cả các biến có thể trả về
        output_map = {
            'b': b,
            'c': c,
            'y': y,
            'x': x,
            'h': h
        }

        return tuple(output_map[item] for item in return_items)
    

    def rec(self, y, snr_ = 1e3):
        no_ = tf.math.reduce_variance(y, axis=[-1,-2,-3,-4])/(snr_ + 1)
        h_hat, err_var = self.chEst(y, no_)
        # print(h_hat.shape)
        llr_det = self.det(y, h_hat, err_var, no_)
        # print(llr_det.shape)
        llr_layer = self.lDemapper(llr_det)
        # print(llr_layer.shape)
        b_hat, tb_crc_status = self.tbDec(llr_layer)
        # print(b_hat.shape)
        return h_hat, llr_det, b_hat, tb_crc_status

    def per(self, y, h, no):       
        h_hat, err_var = h, 0.
        llr_det = self.det(y, h_hat, err_var, no)
        llr_layer = self.lDemapper(llr_det)
        b_hat, tb_crc_status = self.tbDec(llr_layer)

        return h_hat, llr_det, b_hat, tb_crc_status
    
    def clone(self, deep=True):
        """Returns a copy of the Config object

        Input
        -----
        deep : `bool`, (default `True`)
            If `True`, a deep copy will be returned.
        """

        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)