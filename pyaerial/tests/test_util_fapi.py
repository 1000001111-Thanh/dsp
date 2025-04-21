# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Unit tests for aerial/util/fapi.py."""
from aerial.util.fapi import (
    dmrs_fapi_to_bit_array,
    dmrs_bit_array_to_fapi,
    dmrs_fapi_to_sym,
    bit_array_to_mac_pdu,
    mac_pdu_to_bit_array,
)


def test_dmrs_fapi_to_bit_array():
    """Test dmrs_fapi_to_bit_array()."""
    ul_dmrs_symb_pos = 4
    array = dmrs_fapi_to_bit_array(ul_dmrs_symb_pos)
    assert array == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_dmrs_bit_array_to_fapi():
    """Test dmrs_bit_array_to_fapi()."""
    ul_dmrs_symb_pos = 4
    array = dmrs_fapi_to_bit_array(ul_dmrs_symb_pos)
    symb_pos = dmrs_bit_array_to_fapi(array)
    assert ul_dmrs_symb_pos == symb_pos


def test_dmrs_fapi_to_sym():
    """Test dmrs_fapi_to_sym()."""
    ul_dmrs_symb_pos = 4
    symb_idx = dmrs_fapi_to_sym(ul_dmrs_symb_pos)
    assert symb_idx == [2]


def test_mac_pdu_to_bit_array():
    """Test mac_pdu_to_bit_array()."""
    mac_pdu = [32, 4, 16]
    bit_array = mac_pdu_to_bit_array(mac_pdu)
    assert bit_array == [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]


def test_bit_array_to_mac_pdu():
    """Test bit_array_to_mac_pdu()."""
    bit_array = [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    mac_pdu = bit_array_to_mac_pdu(bit_array)
    assert mac_pdu == [32, 192, 7]
