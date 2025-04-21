# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Tests for pusch.py."""
import pytest
import pandas as pd

from aerial.util.data import PuschRecord


def test_pusch_record_simple():

    try:
        df = pd.read_parquet("test_data/example.parquet")
    except FileNotFoundError:
        pytest.skip("Test data not available, skipping...")
        return

    # Grab first row
    row = df.iloc[0]
    dict_row = dict(row)

    # Verify record conversion works
    PuschRecord(**dict_row)

    # Test also the other way of doing the conversion.
    PuschRecord.from_series(row)

    # Test also that columns returns something.
    fields = PuschRecord.columns()
    assert isinstance(fields, tuple)
