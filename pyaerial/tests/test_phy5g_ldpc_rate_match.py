# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Test LdpcRateMatch."""
# Ensure that all the test vectors are available in TEST_VECTOR_DIR.
import glob
import itertools
import pytest
from pytest import TEST_VECTOR_DIR

import h5py as h5
import numpy as np

from aerial.phy5g.ldpc import LdpcRateMatch


# Test vectors are: Different lifting sizes, MCS sweep, some Fujitsu cases
test_case_numbers = [3018, 3019, 3020, 3021, 3022, 3023, 3026, 3027, 3028, 3029, 3030, 3031, 3033,
                     3034, 3035, 3036, 3037, 3038, 3040, 3041, 3042, 3043, 3044, 3046, 3047, 3048,
                     3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3058, 3059, 3060, 3061, 3062,
                     3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075,
                     3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088,
                     3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101,
                     3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114,
                     3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127,
                     3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140,
                     3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153,
                     3154, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918,
                     3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930, 3931,
                     3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944,
                     3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957,
                     3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970,
                     3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979]

all_cases = list(itertools.product(test_case_numbers, [True, False]))


@pytest.mark.parametrize(
    "test_case_number, scrambling",
    all_cases,
    ids=["{}, {}".format(int(test_case_number), scrambling)
         for test_case_number, scrambling in all_cases]
)
def test_ldpc_rate_match(cuda_stream, test_case_number, scrambling):
    """Test LdpcRateMatch."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PDSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    cw_pars = input_file["cw_pars"]
    tbs = 8 * cw_pars["tbSize"][0]
    code_rate = cw_pars["targetCodeRate"][0] / 10240.
    qam_mod_order = cw_pars["qamModOrder"][0]
    rnti = input_file["ue_pars"]["rnti"][0]
    scrambling_id = input_file["ue_pars"]["dataScramId"][0]
    rv = cw_pars["rv"][0]
    num_layers = input_file["ue_pars"]["nUeLayers"][0]
    cinit = (rnti << 15) + scrambling_id

    ref_rate_matched_cbs = np.array(input_file["tb0_ratematcbs"])
    ref_scrambled_cbs = np.array(input_file["tb0_scramcbs"])
    if scrambling:
        ref_output = np.transpose(ref_scrambled_cbs).astype(np.float32)
    else:
        ref_output = np.transpose(ref_rate_matched_cbs).astype(np.float32)

    # Input data.
    coded_cbs = np.array(input_file["tb0_codedcbs"])
    input_data = np.float32(np.transpose(coded_cbs))
    input_data = np.float32(input_data == 1)

    rate_match_len = int(ref_output[:, 0].size)

    ldpc_rate_matcher = LdpcRateMatch(
        enable_scrambling=scrambling,
        cuda_stream=cuda_stream
    )
    rate_matched_bits = ldpc_rate_matcher.rate_match(
        input_data=input_data,
        tb_size=tbs,
        code_rate=code_rate,
        rate_match_len=rate_match_len,
        mod_order=qam_mod_order,
        num_layers=num_layers,
        redundancy_version=rv,
        cinit=cinit
    )

    assert np.array_equal(rate_matched_bits, ref_output)
