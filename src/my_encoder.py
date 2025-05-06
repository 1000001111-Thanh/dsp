#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for LDPC channel encoding and utility functions."""

import tensorflow as tf
import numpy as np
import numbers # to check if n, k are numbers
from sionna.phy.fec.ldpc import LDPC5GEncoder
from sionna.phy.nr.tb_encoder import TBEncoder
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.scrambling import TB5GScrambler
from sionna.phy.nr.utils import calculate_tb_size
from .my_timer import tic, toc

class MyLDPC5GEncoder(LDPC5GEncoder):
    # pylint: disable=line-too-long
    """5G NR LDPC Encoder following the 3GPP 38.212 including rate-matching.

    The implementation follows the 3GPP NR Initiative [3GPPTS38212_LDPC]_.

    Parameters
    ----------
    k: int
        Defining the number of information bit per codeword.

    n: int
        Defining the desired codeword length.

    num_bits_per_symbol: `None` (default) | int
        Defining the number of bits per QAM symbol. If this parameter is
        explicitly provided, the codeword will be interleaved after
        rate-matching as specified in Sec. 5.4.2.2 in [3GPPTS38212_LDPC]_.

    bg: `None` (default) | "bg1" | "bg2"
        Basegraph to be used for the code construction.
        If `None` is provided, the encoder will automatically select
        the basegraph according to [3GPPTS38212_LDPC]_.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    bits: [...,k], tf.float
        Binary tensor containing the information bits to be encoded.

    Output
    ------
    : [...,n], tf.float
        Binary tensor of same shape as inputs besides last dimension has
        changed to `n` containing the encoded codeword bits.

    Note
    ----
    As specified in [3GPPTS38212_LDPC]_, the encoder also performs
    rate-matching (puncturing and shortening). Thus, the corresponding
    decoder needs to `invert` these operations, i.e., must be compatible with
    the 5G encoding scheme.
    """
    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol=None,
                 bg=None,
                 precision=None,
                 **kwargs):

        super(LDPC5GEncoder, self).__init__(precision=precision, **kwargs)

        if not isinstance(k, numbers.Number):
            raise TypeError("k must be a number.")
        if not isinstance(n, numbers.Number):
            raise TypeError("n must be a number.")
        k = int(k) # k or n can be float (e.g. as result of n=k*r)
        n = int(n) # k or n can be float (e.g. as result of n=k*r)

        if k>8448:
            raise ValueError("Unsupported code length (k too large).")
        if k<12:
            raise ValueError("Unsupported code length (k too small).")

        if n>(316*384):
            raise ValueError("Unsupported code length (n too large).")
        if n<0:
            raise ValueError("Unsupported code length (n negative).")

        # init encoder parameters
        self._k = k # number of input bits (= input shape)
        self._n = n # the desired length (= output shape)
        self._coderate = k / n
        self._check_input = True # check input for consistency (i.e., binary)

        # allow actual code rates slightly larger than 948/1024
        # to account for the quantization procedure in 38.214 5.1.3.1
        if self._coderate>(948/1024): # as specified in 38.212 5.4.2.1
            print(f"Warning: effective coderate r>948/1024 for n={n}, k={k}.")
        if self._coderate>(0.95): # as specified in 38.212 5.4.2.1
            raise ValueError(f"Unsupported coderate (r>0.95) for n={n}, k={k}.")

        # construct the basegraph according to 38.212
        # if bg is explicitly provided
        self._bg = self._sel_basegraph(self._k, self._coderate, bg)

        self._z, self._i_ls, self._k_b = self._sel_lifting(self._k, self._bg)
        self._bm = self._load_basegraph(self._i_ls, self._bg)

        # total number of codeword bits
        self._n_ldpc = self._bm.shape[1] * self._z
        # if K_real < K _target puncturing must be applied earlier
        self._k_ldpc = self._k_b * self._z

        # construct explicit graph via lifting
        pcm = self._lift_basegraph(self._bm, self._z)

        pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = self._gen_submat(self._bm,
                                                            self._k_b,
                                                            self._z,
                                                            self._bg)

        # init sub-matrices for fast encoding ("RU"-method)
        # note: dtype is tf.float32;
        self._pcm = pcm # store the sparse parity-check matrix (for decoding)

        # store indices for fast gathering (instead of explicit matmul)
        self._pcm_a_ind = self._mat_to_ind(pcm_a)
        self._pcm_b_inv_ind = self._mat_to_ind(pcm_b_inv)
        self._pcm_c1_ind = self._mat_to_ind(pcm_c1)
        self._pcm_c2_ind = self._mat_to_ind(pcm_c2)

        self._num_bits_per_symbol = num_bits_per_symbol
        if num_bits_per_symbol is not None:
            self._out_int, self._out_int_inv  = self.generate_out_int(self._n,
                                                    self._num_bits_per_symbol)

    def _sel_basegraph(self, k, r, bg_=None):
        """Select basegraph according to [3GPPTS38212_LDPC]_ and check for consistency."""

        # if bg is explicitly provided, we only check for consistency
        if bg_ is None:
            if k <= 292:
                bg = "bg2"
            elif k <= 3824 and r <= 0.67:
                bg = "bg2"
            elif r <= 0.25:
                bg = "bg2"
            else:
                bg = "bg1"
        elif bg_ in ("bg1", "bg2"):
            bg = bg_
        else:
            raise ValueError("Basegraph must be bg1, bg2 or None.")

        # check for consistency
        if bg=="bg1" and k>8448:
            raise ValueError("K is not supported by BG1 (too large).")

        if bg=="bg2" and k>3840:
            raise ValueError(
                f"K is not supported by BG2 (too large) k ={k}.")

        return bg

    def _encode_fast(self, s):
        """Main encoding function based on gathering function."""
        p_a = self._matmul_gather(self._pcm_a_ind, s)
        p_a = self._matmul_gather(self._pcm_b_inv_ind, p_a)
        # calc second part of parity bits p_b
        # second parities are given by C_1*s' + C_2*p_a' + p_b' = 0
        p_b_1 = self._matmul_gather(self._pcm_c1_ind, s)
        p_b_2 = self._matmul_gather(self._pcm_c2_ind, p_a)
        p_b = p_b_1 + p_b_2

        w = tf.concat([p_a, p_b], 1)
        w = tf.cast(w, tf.uint8)
        w = tf.bitwise.bitwise_and(w, tf.constant(1, tf.uint8))
        w = tf.cast(w, self.rdtype)

        c = tf.concat([s, w], 1)
        return c
    
    def call(self, bits):
        """5G LDPC encoding function including rate-matching.

        This function returns the encoded codewords as specified by the 3GPP NR Initiative [3GPPTS38212_LDPC]_ including puncturing and shortening.

        Args:

        bits (tf.float): Tensor of shape `[...,k]` containing the
                information bits to be encoded.

        Returns:

        `tf.float`: Tensor of shape `[...,n]`.
        """

        # Reshape inputs to [...,k]
        input_shape = bits.get_shape().as_list()
        new_shape = [-1, input_shape[-1]]
        u = tf.reshape(bits, new_shape)

        # assert if bits are non-binary
        if self._check_input:
            tf.debugging.assert_equal(
                tf.reduce_min(
                    tf.cast(
                        tf.logical_or(
                            tf.equal(u, tf.constant(0, self.rdtype)),
                            tf.equal(u, tf.constant(1, self.rdtype)),
                            ),
                        self.rdtype)),
                tf.constant(1, self.rdtype),
                "Input must be binary.")
            # input datatype consistency should be only evaluated once
            self._check_input = False

        batch_size = tf.shape(u)[0]

        # add "filler" bits to last positions to match info bit length k_ldpc
        u_fill = tf.concat([u,
            tf.zeros([batch_size, self._k_ldpc-self._k], self.rdtype)],axis=1)

        # use optimized encoding based on tf.gather
        c = self._encode_fast(u_fill)

        c = tf.reshape(c, [batch_size, self._n_ldpc]) # remove last dim

        c_short_2z = tf.slice(c, [0, 2*self._z], [batch_size,  self._n_ldpc-2*self._z])

        c_no_filler1 = tf.slice(c_short_2z, [0, 0], [batch_size, self._k-2*self._z])

        size_no_filler2 = min(self._n_ldpc-self._k_ldpc, c_short_2z.shape[1])
        c_no_filler2 = tf.slice(c_short_2z,
                               [0, self._k_ldpc-2*self._z],
                               [batch_size, size_no_filler2])
        
        c_no_filler = tf.concat([c_no_filler1, c_no_filler2], 1)
        if c_no_filler.shape[1] < self._n:
            c_no_filler = tf.tile(c_no_filler, [1, (self._n / c_no_filler.shape[1]).__ceil__()]) 

        c_short = tf.slice(c_no_filler, [0, 0], [batch_size,  self._n])
        # incremental redundancy could be generated by accessing the last bits

        # if num_bits_per_symbol is provided, apply output interleaver as
        # specified in Sec. 5.4.2.2 in 38.212
        if self._num_bits_per_symbol is not None:
            c_short = tf.gather(c_short, self._out_int, axis=-1)

        # Reshape c_short so that it matches the original input dimensions
        output_shape = input_shape[0:-1] + [self.n]
        output_shape[0] = -1
        c_reshaped = tf.reshape(c_short, output_shape)

        return c_reshaped



class MyTB5GScrambler(TB5GScrambler):
    # pylint: disable=line-too-long
    r"""5G NR Scrambler for PUSCH and PDSCH channel.

    Implements the pseudo-random bit scrambling as defined in
    [3GPPTS38211_scr]_ Sec. 6.3.1.1 for the "PUSCH" channel and in Sec. 7.3.1.1
    for the "PDSCH" channel.

    Only for the "PDSCH" channel, the scrambler can be configured for two
    codeword transmission mode. Hereby, ``codeword_index`` corresponds to the
    index of the codeword to be scrambled.

    If ``n_rnti`` are a list of ints, the scrambler assumes that the second
    last axis contains `len(` ``n_rnti`` `)` elements. This allows independent
    scrambling for multiple independent streams.

    Parameters
    ----------
    n_rnti: int | list of ints
        RNTI identifier provided by higher layer. Defaults to 1 and must be
        in range `[0, 65335]`. If a list is provided, every list element
        defines a scrambling sequence for multiple independent streams.

    n_id: int | list of ints
        Scrambling ID related to cell id and provided by higher layer.
        Defaults to 1 and must be in range `[0, 1023]`. If a list is
        provided, every list element defines a scrambling sequence for
        multiple independent streams.

    binary: `bool`, (default `True`)
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    channel_type: str, 'PUSCH' | 'PDSCH'
        Can be either 'PUSCH' or 'PDSCH'.

    codeword_index: int, 0 | 1
        Scrambler can be configured for two codeword transmission.
        ``codeword_index`` can be either 0 or 1.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: tf.float
        Tensor of arbitrary shape. If ``n_rnti`` and ``n_id`` are a
        list, it is assumed that ``x`` has shape
        `[...,num_streams, n]` where `num_streams=len(` ``n_rnti`` `)`.

    binary: `None` (default) | bool
        Overrules the init parameter `binary` iff explicitly given.
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    Output
    ------
    : tf.float
        Tensor of same shape as ``x``.

    Note
    ----
    The parameters radio network temporary identifier (RNTI) ``n_rnti`` and
    the datascrambling ID ``n_id`` are usually provided be the higher layer protocols.

    For inverse scrambling, the same scrambler can be re-used (as the values
    are flipped again, i.e., result in the original state).
    """

    #################
    # Utility methods
    #################

    # pylint: disable=(unused-argument)
    def build(self, input_shape, **kwargs):
        """Initialize pseudo-random scrambling sequence."""

        self._input_shape = input_shape

        # in multi-stream mode, the axis=-2 must have dimension=len(c_init)
        if self._multi_stream:
            assert input_shape[-2]==len(self._c_init), \
                "Dimension of axis=-2 must be equal to len(n_rnti)."

        self._sequence = tf.Variable(self._generate_scrambling(input_shape), dtype=self.rdtype)
    
    @property
    def c_init(self):
        return self._c_init
    
    @c_init.setter
    def c_init(self, v):
        tic()
        if not isinstance(v, (list, tuple)):
            v = [v]
        self._c_init = v
        toc("assgin c_init with v")
        tic()
        self.sequence = self._generate_scrambling(self._input_shape)
        toc("generate_scrambling")

    @property
    def sequence(self):
        return self._sequence
    
    @sequence.setter
    def sequence(self, v):
        self._sequence.assign(self._cast_or_check_precision(v))

    @tf.function
    def call(self, x, /, *, binary=None):
        r"""This function returns the scrambled version of ``x``.
        """

        if binary is None:
            binary = self._binary
        else:
            # allow tf.bool as well
            if not (isinstance(binary, bool) or \
                (tf.is_tensor(binary) and binary.dtype == tf.bool)):
                raise TypeError("binary must be bool.")

        # if not x.shape[-1]==self._input_shape:
        #     self.build(x.shape)

        # support various non-float dtypes
        input_dtype = x.dtype
        x = tf.cast(x, self.rdtype)

        if binary:
            # flip the bits by subtraction and map -1 to 1 via abs(.) operator
            x_out = tf.abs(x - self.sequence)
        else:
            rand_seq_bipol = -2 * self.sequence + 1
            x_out = tf.multiply(x, rand_seq_bipol)

        return tf.cast(x_out, input_dtype)   

class MyTBEncoder(TBEncoder):
    # pylint: disable=line-too-long
    r"""5G NR transport block (TB) encoder as defined in TS 38.214
    [3GPP38214]_ and TS 38.211 [3GPP38211]_

    The transport block (TB) encoder takes as input a `transport block` of
    information bits and generates a sequence of codewords for transmission.
    For this, the information bit sequence is segmented into multiple codewords,
    protected by additional CRC checks and FEC encoded. Further, interleaving
    and scrambling is applied before a codeword concatenation generates the
    final bit sequence. Fig. 1 provides an overview of the TB encoding
    procedure and we refer the interested reader to [3GPP38214]_ and
    [3GPP38211]_ for further details.

    ..  figure:: ../figures/tb_encoding.png

        Fig. 1: Overview TB encoding (CB CRC does not always apply).

    If ``n_rnti`` and ``n_id`` are given as list, the TBEncoder encodes
    `num_tx = len(` ``n_rnti`` `)` parallel input streams with different
    scrambling sequences per user.

    Parameters
    ----------
    target_tb_size: `int`
        Target transport block size, i.e., how many information bits are
        encoded into the TB. Note that the effective TB size can be
        slightly different due to quantization. If required, zero padding
        is internally applied.

    num_coded_bits: `int`
        Number of coded bits after TB encoding

    target_coderate : `float`
        Target coderate

    num_bits_per_symbol: `int`
        Modulation order, i.e., number of bits per QAM symbol

    num_layers: 1 (default) | [1,...,8]
        Number of transmission layers

    n_rnti: `int` or `list` of `int`, 1 (default) | [0,...,65335]
        RNTI identifier provided by higher layer. Defaults to 1 and must be
        in range `[0, 65335]`. Defines a part of the random seed of the
        scrambler. If provided as list, every list entry defines the RNTI
        of an independent input stream.

    n_id: `int` or `list` of `int`, 1 (default) | [0,...,1023]
        Data scrambling ID :math:`n_\text{ID}` related to cell id and
        provided by higher layer.
        Defaults to 1 and must be in range `[0, 1023]`. If provided as
        list, every list entry defines the scrambling id of an independent
        input stream.

    channel_type: "PUSCH" (default) | "PDSCH"
        Can be either "PUSCH" or "PDSCH".

    codeword_index: 0 (default) | 1
        Scrambler can be configured for two codeword transmission.
        ``codeword_index`` can be either 0 or 1. Must be 0 for
        ``channel_type`` = "PUSCH".

    use_scrambler: bool, (default `True`)
        If False, no data scrambling is applied (non standard-compliant).

    verbose: bool, (default `False`)
        If `True`, additional parameters are printed during initialization.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    inputs: [...,target_tb_size] or [...,num_tx,target_tb_size], tf.float
        2+D tensor containing the information bits to be encoded. If
        ``n_rnti`` and ``n_id`` are a list of size `num_tx`, the input must
        be of shape `[...,num_tx,target_tb_size]`.

    Output
    ------
    : [...,num_coded_bits], tf.float
        2+D tensor containing the sequence of the encoded codeword bits of
        the transport block.

    Note
    ----
    The parameters ``tb_size`` and ``num_coded_bits`` can be derived by the
    :meth:`~sionna.phy.nr.calculate_tb_size` function or
    by accessing the corresponding :class:`~sionna.phy.nr.PUSCHConfig` attributes.
    """

    def __init__(self,
                 target_tb_size,
                 num_coded_bits,
                 target_coderate,
                 num_bits_per_symbol,
                 num_layers=1,
                 n_rnti=1,
                 n_id=1,
                 channel_type="PUSCH",
                 codeword_index=0,
                 use_scrambler=True,
                 verbose=False,
                 precision=None,
                 **kwargs):

        super(TBEncoder, self).__init__(precision=precision, **kwargs)

        assert isinstance(use_scrambler, bool), \
                                "use_scrambler must be bool."
        self._use_scrambler = use_scrambler
        assert isinstance(verbose, bool), \
                                "verbose must be bool."
        self._verbose = verbose

        # check input for consistency
        assert channel_type in ("PDSCH", "PUSCH"), \
                                "Unsupported channel_type."
        self._channel_type = channel_type

        assert(target_tb_size%1==0), "target_tb_size must be int."
        self._target_tb_size = int(target_tb_size)

        assert(num_coded_bits%1==0), "num_coded_bits must be int."
        self._num_coded_bits = int(num_coded_bits)

        assert(0.<target_coderate <= 948/1024), \
                    "target_coderate must be in range(0,0.925)."
        self._target_coderate = target_coderate

        assert(num_bits_per_symbol%1==0), "num_bits_per_symbol must be int."
        self._num_bits_per_symbol = int(num_bits_per_symbol)

        assert(num_layers%1==0), "num_layers must be int."
        self._num_layers = int(num_layers)

        if channel_type=="PDSCH":
            assert(codeword_index in (0,1)), "codeword_index must be 0 or 1."
        else:
            assert codeword_index==0, 'codeword_index must be 0 for "PUSCH".'
        self._codeword_index = int(codeword_index)

        if isinstance(n_rnti, (list, tuple)):
            assert isinstance(n_id, (list, tuple)), "n_id must be also a list."
            assert (len(n_rnti)==len(n_id)), \
                                "n_id and n_rnti must be of same length."
            self._n_rnti = n_rnti
            self._n_id = n_id
        else:
            self._n_rnti = [n_rnti]
            self._n_id = [n_id]

        for idx, n in enumerate(self._n_rnti):
            assert(n%1==0), "n_rnti must be int."
            self._n_rnti[idx] = int(n)
        for idx, n in enumerate(self._n_id):
            assert(n%1==0), "n_id must be int."
            self._n_id[idx] = int(n)

        self._num_tx = len(self._n_id)

        tbconfig = calculate_tb_size(target_tb_size=self._target_tb_size,
                                     num_coded_bits=self._num_coded_bits,
                                     target_coderate=self._target_coderate,
                                     modulation_order=self._num_bits_per_symbol,
                                     num_layers=self._num_layers,
                                     verbose=verbose)
        self._tb_size = tbconfig[0].numpy()
        self._cb_size = tbconfig[1].numpy()
        self._num_cbs = tbconfig[2].numpy()
        self._tb_crc_length = tbconfig[3].numpy()
        self._cb_crc_length = tbconfig[4].numpy()
        self._cw_lengths = tbconfig[5].numpy()

        assert self._tb_size <= self._tb_crc_length + np.sum(self._cw_lengths),\
            "Invalid TB parameters."

        # due to quantization, the tb_size can slightly differ from the
        # target tb_size.
        self._k_padding = self._tb_size - self._target_tb_size
        if self._tb_size != self._target_tb_size:
            print(f"Note: actual tb_size={self._tb_size} is slightly "\
                  f"different than requested " \
                  f"target_tb_size={self._target_tb_size} due to "\
                  f"quantization. Internal zero padding will be applied.")

        # calculate effective coderate (incl. CRC)
        self._coderate = self._tb_size / self._num_coded_bits

        # Remark: CRC16 is only used for k<3824 (otherwise CRC24)
        if self._tb_crc_length==16:
            self._tb_crc_encoder = CRCEncoder("CRC16", precision=precision)
        else:
            # CRC24A as defined in 7.2.1
            self._tb_crc_encoder = CRCEncoder("CRC24A", precision=precision)

        # CB CRC only if more than one CB is used
        if self._cb_crc_length==24:
            self._cb_crc_encoder = CRCEncoder("CRC24B", precision=precision)
        else:
            self._cb_crc_encoder = None

        # scrambler can be deactivated (non-standard compliant)
        if self._use_scrambler:
            self._scrambler = MyTB5GScrambler(n_rnti=self._n_rnti,
                                            n_id=self._n_id,
                                            binary=True,
                                            channel_type=channel_type,
                                            codeword_index=codeword_index,
                                            precision=precision)
            input_shape = (None, len(self._n_rnti), np.sum(self._cw_lengths))
            self._scrambler.build(input_shape)
        else: # required for TBDecoder
            self._scrambler = None

        # ---- Init LDPC encoder ----
        # remark: as the codeword length can be (slightly) different
        # within a TB due to rounding, we initialize the encoder
        # with the max length and apply puncturing if required.
        # Thus, also the output interleaver cannot be applied in the encoder.
        # The procedure is defined in in 5.4.2.1 38.212
        self._encoder = MyLDPC5GEncoder(self._cb_size,
                                      np.max(self._cw_lengths),
                                      num_bits_per_symbol=1,
                                      precision=precision) #deact. interleaver

        # ---- Init interleaver ----
        # remark: explicit interleaver is required as the rate matching from
        # Sec. 5.4.2.1 38.212 could otherwise not be applied here
        perm_seq_short, _ = self._encoder.generate_out_int(
                                            np.min(self._cw_lengths),
                                            num_bits_per_symbol)
        perm_seq_long, _ = self._encoder.generate_out_int(
                                            np.max(self._cw_lengths),
                                            num_bits_per_symbol)

        perm_seq = []
        perm_seq_punc = []

        # define one big interleaver that moves the punctured positions to the
        # end of the TB
        payload_bit_pos = 0 # points to current pos of payload bits

        for l in self._cw_lengths:
            if np.min(self._cw_lengths)==l:
                perm_seq = np.concatenate([perm_seq,
                                           perm_seq_short + payload_bit_pos])
                # move unused bit positions to the end of TB
                # this simplifies the inverse permutation
                r = np.arange(payload_bit_pos+np.min(self._cw_lengths),
                              payload_bit_pos+np.max(self._cw_lengths))
                perm_seq_punc = np.concatenate([perm_seq_punc, r])

                # update pointer
                payload_bit_pos += np.max(self._cw_lengths)
            elif np.max(self._cw_lengths)==l:
                perm_seq = np.concatenate([perm_seq,
                                           perm_seq_long + payload_bit_pos])
                # update pointer
                payload_bit_pos += l
            else:
                raise ValueError("Invalid cw_lengths.")

        # add punctured positions to end of sequence (only relevant for
        # deinterleaving)
        perm_seq = np.concatenate([perm_seq, perm_seq_punc])

        self._output_perm = tf.constant(perm_seq, tf.int32)
        self._output_perm_inv = tf.argsort(perm_seq, axis=-1)

