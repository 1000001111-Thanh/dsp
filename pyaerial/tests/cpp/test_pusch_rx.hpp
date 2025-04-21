/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_TEST_PUSCH_RX_HPP
#define PYCUPHY_TEST_PUSCH_RX_HPP

#include "pycuphy_params.hpp"
#include "pycuphy_channel_est.hpp"
#include "pycuphy_channel_eq.hpp"
#include "pycuphy_noise_intf_est.hpp"
#include "pycuphy_rsrp.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_crc_check.hpp"

namespace pycuphy {

class TestPuschRxPipeline {
public:
    TestPuschRxPipeline(PuschParams& puschParams, const cudaStream_t cudaStream);

    bool runTest(PuschParams& puschParams, std::string& errMsg);

private:

    cudaStream_t m_cudaStream;

    // PUSCH Rx pipeline components
    ChannelEstimator    m_chEstimator;
    NoiseIntfEstimator  m_noiseIntfEstimator;
    ChannelEqualizer    m_chEqualizer;
    RsrpEstimator       m_rsrpEstimator;
    LdpcDerateMatch     m_derateMatch;
    LdpcDecoder         m_decoder;
    CrcChecker          m_crcChecker;
};

} // namespace pycuphy

#endif // PYCUPHY_TEST_PUSCH_RX_HPP