from dataclasses import dataclass, field
from typing import List
import numpy as np

CARRIER_FREQUENCY = 2.55e9
BANDWIDTH = 60
NUM_RX = 1
NUM_TX = 1
NUM_RX_ANT = 4
NUM_TX_ANT = 1
NUM_STREAMS_PER_TX = 1

NUM_PRB_MAX = 162
NUM_SYMBOLS_PER_SLOT = 14

@dataclass
class SystemConfig:
    NCellId: np.uint16 = 0
    FrequencyRange: np.uint8 =  1
    BandWidth: np.uint16 = BANDWIDTH
    Numerology: np.uint8 = 1
    CpType: np.uint8 = 0
    NTxAnt: np.uint8 = NUM_TX_ANT
    NRxAnt: np.uint8 = NUM_RX_ANT
    BwpNRb: np.uint16 = NUM_PRB_MAX
    BwpRbOffset: np.uint16 = 0
    harqProcFlag: np.uint8 = 0
    nHarqProc: np.uint8 = 1
    rvSeq: np.uint8 = 0


@dataclass
class UeConfig:
    TransformPrecoding: np.uint8 = 0
    Rnti: np.uint16 = 20002
    nId: np.uint16 = None
    CodeBookBased: np.uint8 = 0
    DmrsPortSetIdx: List[np.uint8] = field(default_factory=lambda: [0])  # FIXED
    NLayers: np.uint8 = 1
    NumDmrsCdmGroupsWithoutData: np.uint8 = 2
    Tpmi: np.uint8 = 0
    FirstSymb: np.uint8 = 0
    NPuschSymbAll: np.uint8 = NUM_SYMBOLS_PER_SLOT
    RaType: np.uint8 = 1
    FirstPrb: np.uint16 = 0
    NPrb: np.uint16 = NUM_PRB_MAX
    FrequencyHoppingMode: np.uint8 = 0
    McsTable: np.uint8 = 0
    Mcs: np.uint8 = 0
    ILbrm: np.uint8 = 0
    nScId: np.uint8 = 0
    NnScIdId: np.uint8 = None
    DmrsConfigurationType: np.uint8 = 0
    DmrsDuration: np.uint8 = 1
    DmrsAdditionalPosition: np.uint8 = 1
    PuschMappingType: np.uint8 = 0
    DmrsTypeAPosition: np.uint8 = 3
    HoppingMode: np.uint8 = 0
    NRsId: np.uint8 = 0
    Ptrs: np.uint8 = 0
    ScalingFactor: np.uint8 = 0
    OAck: np.uint8 = 0
    IHarqAckOffset: np.uint8 = 11
    OCsi1: np.uint8 = 0
    ICsi1Offset: np.uint8 = 7
    OCsi2: np.uint8 = 0
    ICsi2Offset: np.uint8 = 0
    NPrbOh: np.uint8 = 0
    nCw: np.uint8 = 1
    TpPi2Bpsk: np.uint8 = 0

@dataclass
class MyConfig:
    Sys: SystemConfig
    Ue: UeConfig
    Carrier_frequency: float = CARRIER_FREQUENCY  # Carrier frequency in Hz
