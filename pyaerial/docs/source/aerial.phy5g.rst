pyAerial - Physical layer pipelines for 5G
==========================================
This module contains classes implementing the 5G NR physical layer using
GPU acceleration through the cuPHY library. The module contains full PDSCH transmitter
and PUSCH receiver pipelines in :class:`~aerial.phy5g.pdsch.pdsch_tx.PdschTx` and
:class:`~aerial.phy5g.pusch.pusch_rx.PuschRx`, respectively. The other parts of this module contain
individual components of the transmitter-receiver chain, such as for example the LDPC
encoder and decoder in :class:`~aerial.phy5g.ldpc.ldpc_encoder.LdpcEncoder` and
:class:`~aerial.phy5g.ldpc.ldpc_decoder.LdpcDecoder`, and the channel estimator in
:class:`~aerial.phy5g.algorithms.channel_estimator.ChannelEstimator`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   aerial.phy5g.algorithms
   aerial.phy5g.pdsch
   aerial.phy5g.pusch
   aerial.phy5g.ldpc
