import tensorflow as tf
import numpy as np
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.nr.tb_decoder import TBDecoder, TBEncoder
from sionna.phy.fec.crc import CRCDecoder
from sionna.phy.fec.scrambling import  Descrambler
from sionna.phy.fec.ldpc import LDPC5GDecoder

class MyLDPC5GDecoder(LDPC5GDecoder):
    # pylint: disable=line-too-long
    r"""Iterative belief propagation decoder for 5G NR LDPC codes.

    Inherits from :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and
    provides a wrapper for 5G compatibility, i.e., automatically handles
    rate-matching according to [3GPPTS38212_LDPC]_.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied and, thus, the encoder object is
    required as input.

    If required the decoder can be made trainable and is differentiable
    (the training of some check node types may be not supported) following the
    concept of "weighted BP" [Nachmani]_.

    Parameters
    ----------
    encoder: LDPC5GEncoder
        An instance of :class:`~sionna.phy.fec.ldpc.encoding.LDPC5GEncoder`
        containing the correct code parameters.

    cn_update: `str`, "boxplus-phi" (default) | "boxplus" | "minsum" | "offset-minsum" | "identity" | callable
        Check node update rule to be used as described above.
        If a callable is provided, it will be used instead as CN update.
        The input of the function is a ragged tensor of v2c messages of shape
        `[num_cns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual CN degree).

    vn_update: `str`, "sum" (default) | "identity" | callable
        Variable node update rule to be used.
        If a callable is provided, it will be used instead as VN update.
        The input of the function is a ragged tensor of c2v messages of shape
        `[num_vns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual VN degree).

    cn_schedule: "flooding" | "layered" | [num_update_steps, num_active_nodes], tf.int
        Defines the CN update scheduling per BP iteration. Can be either
        "flooding" to update all nodes in parallel (recommended) or "layered"
        to sequentally update all CNs in the same lifting group together or an
        2D tensor where each row defines the `num_active_nodes` node indices to
        be updated per subiteration. In this case each BP iteration runs
        `num_update_steps` subiterations, thus the decoder's level of
        parallelization is lower and usually the decoding throughput decreases.

    hard_out: `bool`, (default `True`)
        If `True`,  the decoder provides hard-decided codeword bits instead of
        soft-values.

    return_infobits: `bool`, (default `True`)
        If `True`, only the `k` info bits (soft or hard-decided) are returned.
        Otherwise all `n` positions are returned.

    prune_pcm: `bool`, (default `True`)
        If `True`, all punctured degree-1 VNs and connected check nodes are
        removed from the decoding graph (see [Cammerer]_ for details). Besides
        numerical differences, this should yield the same decoding result but
        improved the decoding throughput and reduces the memory footprint.

    num_iter: `int` (default: 20)
        Defining the number of decoder iterations (due to batching, no early
        stopping used at the moment!).

    llr_max: `float` (default: 20) | `None`
        Internal clipping value for all internal messages. If `None`, no
        clipping is applied.

    v2c_callbacks, `None` (default) | list of callables
        Each callable will be executed after each VN update with the following
        arguments `msg_vn_rag_`, `it`, `x_hat`,where `msg_vn_rag_` are the v2c
        messages as ragged tensor of shape `[num_vns, None, batch_size]`,
        `x_hat` is the current estimate of each VN of shape
        `[num_vns, batch_size]` and `it` is the current iteration counter.
        It must return and updated version of `msg_vn_rag_` of same shape.

    c2v_callbacks: `None` (default) | list of callables
        Each callable will be executed after each CN update with the following
        arguments `msg_cn_rag_` and `it` where `msg_cn_rag_` are the c2v
        messages as ragged tensor of shape `[num_cns, None, batch_size]` and
        `it` is the current iteration counter.
        It must return and updated version of `msg_cn_rag_` of same shape.

    return_state: `bool`, (default `False`)
        If `True`,  the internal VN messages ``msg_vn`` from the last decoding
        iteration are returned, and ``msg_vn`` or `None` needs to be given as a
        second input when calling the decoder.
        This can be used for iterative demapping and decoding.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    llr_ch: [...,n], tf.float
        Tensor containing the channel logits/llr values.

    msg_v2c: `None` | [num_edges, batch_size], tf.float
        Tensor of VN messages representing the internal decoder state.
        Required only if the decoder shall use its previous internal state, e.g.
        for iterative detection and decoding (IDD) schemes.

    Output
    ------
    : [...,n] or [...,k], tf.float
        Tensor of same shape as ``llr_ch`` containing
        bit-wise soft-estimates (or hard-decided bit-values) of all
        `n` codeword bits or only the `k` information bits if
        ``return_infobits`` is True.

    : [num_edges, batch_size], tf.float:
        Tensor of VN messages representing the internal decoder state.
        Returned only if ``return_state`` is set to `True`.
        Remark: always retruns entire decoder state, even if
        ``return_infobits`` is True.

    Note
    ----
    As decoding input logits :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}`
    are assumed for compatibility with the learning framework, but internally
    LLRs with definition :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are
    used.

    The decoder is not (particularly) optimized for Quasi-cyclic (QC) LDPC
    codes and, thus, supports arbitrary parity-check matrices.

    The decoder is implemented by using '"ragged Tensors"' [TF_ragged]_ to
    account for arbitrary node degrees. To avoid a performance degradation
    caused by a severe indexing overhead, the batch-dimension is shifted to
    the last dimension during decoding.

    """

    def __init__(self,
                 encoder,
                 cn_update="boxplus-phi",
                 vn_update="sum",
                 cn_schedule="flooding",
                 hard_out=True,
                 return_infobits=True,
                 num_iter=20,
                 llr_max=20.,
                 v2c_callbacks=None,
                 c2v_callbacks=None,
                 prune_pcm=True,
                 return_state=False,
                 precision=None,
                 **kwargs):

        # needs the 5G Encoder to access all 5G parameters
        if not isinstance(encoder, LDPC5GEncoder):
            raise TypeError("encoder must be of class LDPC5GEncoder.")

        self._encoder = encoder
        pcm = encoder.pcm

        if not isinstance(return_infobits, bool):
            raise TypeError('return_info must be bool.')
        self._return_infobits = return_infobits

        if not isinstance(return_state, bool):
            raise TypeError('return_state must be bool.')
        self._return_state = return_state

        # Deprecation warning for cn_type
        if 'cn_type' in kwargs:
            raise TypeError("'cn_type' is deprecated; use 'cn_update' instead.")

        # prune punctured degree-1 VNs and connected CNs. A punctured
        # VN-1 node will always "send" llr=0 to the connected CN. Thus, this
        # CN will only send 0 messages to all other VNs, i.e., does not
        # contribute to the decoding process.
        if not isinstance(prune_pcm, bool):
            raise TypeError('prune_pcm must be bool.')
        self._prune_pcm = prune_pcm
        if prune_pcm:
            # find index of first position with only degree-1 VN
            dv = np.sum(pcm, axis=0) # VN degree
            last_pos = encoder._n_ldpc
            for idx in range(encoder._n_ldpc-1, 0, -1):
                if dv[0, idx]==1:
                    last_pos = idx
                else:
                    break
            # number of filler bits
            k_filler = self.encoder.k_ldpc - self.encoder.k

            # number of punctured bits
            nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                                     - self.encoder.n - 2*self.encoder.z)

            # if layered decoding is used, qunatized number of punctured bits
            # to a multiple of z; otherwise scheduling groups of Z CNs becomes
            # impossible
            if cn_schedule=="layered":
                nb_punc_bits = np.floor(nb_punc_bits/self.encoder.z) \
                             * self.encoder.z
                nb_punc_bits = int (nb_punc_bits) # cast to int

            # effective codeword length after pruning of vn-1 nodes
            self._n_pruned = np.max((last_pos, encoder._n_ldpc - nb_punc_bits))
            self._nb_pruned_nodes = max(0, encoder._n_ldpc - self._n_pruned)
            # remove last CNs and VNs from pcm
            if self._nb_pruned_nodes > 0:
                pcm = pcm[:-self._nb_pruned_nodes, :-self._nb_pruned_nodes]
            

            #check for consistency
            if self._nb_pruned_nodes<0:
                msg = "Internal error: number of pruned nodes must be positive."
                raise ArithmeticError(msg)
        else:
            # no pruning; same length as before
            self._nb_pruned_nodes = 0
            self._n_pruned = encoder._n_ldpc

        if cn_schedule=="layered":
            z = self._encoder.z
            num_blocks = int(pcm.shape[0]/z)
            cn_schedule = []
            for i in range(num_blocks):
                cn_schedule.append(np.arange(z) + i*z)
            cn_schedule = tf.stack(cn_schedule, axis=0)

        super(LDPC5GDecoder, self).__init__(pcm,
                         cn_update=cn_update,
                         vn_update=vn_update,
                         cn_schedule=cn_schedule,
                         hard_out=hard_out,
                         num_iter=num_iter,
                         llr_max=llr_max,
                         v2c_callbacks=v2c_callbacks,
                         c2v_callbacks=c2v_callbacks,
                         return_state=return_state,
                         precision=precision,
                         **kwargs)

    ###############################
    # Public methods and properties
    ###############################

    def call(self, llr_ch, /, *, num_iter=None, msg_v2c=None):
        """Iterative BP decoding function and rate matching.
        """
        llr_ch_shape = llr_ch.get_shape().as_list()
        print(llr_ch_shape, self.encoder.n)
        new_shape = [-1, self.encoder.n]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        # invert if rate-matching output interleaver was applied as defined in
        # Sec. 5.4.2.2 in 38.212
        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,
                                        self._encoder.out_int_inv,
                                        axis=-1)

        # undo puncturing of the first 2*Z bit positions
        llr_5g = tf.concat(
                    [tf.zeros([batch_size, 2*self.encoder.z], self.rdtype),
                    llr_ch_reshaped], axis=1)

        # undo puncturing of the last positions
        # total length must be n_ldpc, while llr_ch has length n
        # first 2*z positions are already added
        # -> add n_ldpc - n - 2Z punctured positions
        k_filler = self.encoder.k_ldpc - self.encoder.k # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                        - self.encoder.n - 2*self.encoder.z)
        if nb_punc_bits - self._nb_pruned_nodes > 0:
            llr_5g = tf.concat([llr_5g,
                        tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],
                                self.rdtype)], axis=1)

        # undo shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
        # the first k positions are the systematic bits
        x1 = tf.slice(llr_5g, [0,0], [batch_size, self.encoder.k])

        # parity part
        nb_par_bits = (self.encoder.n_ldpc - k_filler
                       - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,
                      [0, self.encoder.k],
                      [batch_size, nb_par_bits])

        # negative sign due to logit definition
        z = -tf.cast(self._llr_max, self.rdtype) \
            * tf.ones([batch_size, k_filler], self.rdtype)

        llr_5g = tf.concat([x1, z, x2], axis=1)

        # and run the core decoder
        output = super(LDPC5GDecoder, self).call(llr_5g, num_iter=num_iter, msg_v2c=msg_v2c)

        if self._return_state:
            x_hat, msg_v2c = output
        else:
            x_hat = output


        if self._return_infobits:# return only info bits
            # reconstruct u_hat
            # 5G NR code is systematic
            u_hat = tf.slice(x_hat, [0,0], [batch_size, self.encoder.k])
            # Reshape u_hat so that it matches the original input dimensions
            output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
            # overwrite first dimension as this could be None
            output_shape[0] = -1
            u_reshaped = tf.reshape(u_hat, output_shape)

            if self._return_state:
                return u_reshaped, msg_v2c
            else:
                return u_reshaped

        else: # return all codeword bits
            # The transmitted CW bits are not the same as used during decoding
            # cf. last parts of 5G encoding function

            # remove last dim
            x = tf.reshape(x_hat, [batch_size, self._n_pruned])

            # remove filler bits at pos (k, k_ldpc)
            x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])

            x_no_filler2 = tf.slice(x,
                                    [0, self.encoder.k_ldpc],
                                    [batch_size,
                                    self._n_pruned-self.encoder.k_ldpc])

            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

            # shorten the first 2*Z positions and end after n bits
            x_short = tf.slice(x_no_filler,
                               [0, 2*self.encoder.z],
                               [batch_size, self.encoder.n])

            # if used, apply rate-matching output interleaver again as
            # Sec. 5.4.2.2 in 38.212
            if self._encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

            # Reshape x_short so that it matches the original input dimensions
            # overwrite first dimension as this could be None
            llr_ch_shape[0] = -1
            x_short= tf.reshape(x_short, llr_ch_shape)

            if self._return_state:
                return x_short, msg_v2c
            else:
                return x_short

class MyTBDecoder(TBDecoder):
    # pylint: disable=line-too-long
    r"""TBDecoder(encoder, num_bp_iter=20, cn_type="boxplus-phi", precision=None, **kwargs)
    5G NR transport block (TB) decoder as defined in TS 38.214
    [3GPP38214]_.

    The transport block decoder takes as input a sequence of noisy channel
    observations and reconstructs the corresponding `transport block` of
    information bits. The detailed procedure is described in TS 38.214
    [3GPP38214]_ and TS 38.211 [3GPP38211]_.

    Parameters
    ----------
    encoder : :class:`~sionna.phy.nr.TBEncoder`
        Associated transport block encoder used for encoding of the signal

    num_bp_iter : 20 (default) | `int`
        Number of BP decoder iterations

    cn_update: str, "boxplus-phi" (default) | "boxplus" | "minsum" | "offset-minsum" | "identity" | callable
        Check node update rule to be used as described above.
        If a callable is provided, it will be used instead as CN update.
        The input of the function is a ragged tensor of v2c messages of shape
        `[num_cns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual CN degree).

    vn_update: str, "sum" (default) | "identity" | callable
        Variable node update rule to be used.
        If a callable is provided, it will be used instead as VN update.
        The input of the function is a ragged tensor of c2v messages of shape
        `[num_vns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual VN degree).

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    inputs : [...,num_coded_bits], `tf.float`
        2+D tensor containing channel logits/llr values of the (noisy)
        channel observations.

    Output
    ------
    b_hat : [...,target_tb_size], `tf.float`
        2+D tensor containing hard decided bit estimates of all information
        bits of the transport block.

    tb_crc_status : [...], `tf.bool`
        Transport block CRC status indicating if a transport block was
        (most likely) correctly recovered. Note that false positives are
        possible.
    """

    def __init__(self,
                 encoder,
                 num_bp_iter=20,
                 cn_update="boxplus-phi",
                 vn_update="sum",
                 precision=None,
                 **kwargs):

        super(TBDecoder, self).__init__(precision=precision, **kwargs)

        assert isinstance(encoder, TBEncoder), "encoder must be TBEncoder."
        self._tb_encoder = encoder

        self._num_cbs = encoder.num_cbs

        # init BP decoder
        self._decoder = MyLDPC5GDecoder(encoder=encoder.ldpc_encoder,
                                      num_iter=num_bp_iter,
                                      cn_update=cn_update,
                                      vn_update=vn_update,
                                      hard_out=True, # TB operates on bit-level
                                      return_infobits=True,
                                      precision=precision)

        # init descrambler
        if encoder.scrambler is not None:
            self._descrambler = Descrambler(encoder.scrambler,
                                            binary=False,
                                            precision=precision)
        else:
            self._descrambler = None

        # init CRC Decoder for CB and TB
        self._tb_crc_decoder = CRCDecoder(encoder.tb_crc_encoder,
                                          precision=precision)

        if encoder.cb_crc_encoder is not None:
            self._cb_crc_decoder = CRCDecoder(encoder.cb_crc_encoder,
                                              precision=precision)
        else:
            self._cb_crc_decoder = None

