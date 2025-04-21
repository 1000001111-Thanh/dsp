/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include "pycuphy_params.hpp"
#include "pycuphy_channel_est.hpp"
#include "pycuphy_channel_eq.hpp"
#include "pycuphy_noise_intf_est.hpp"
#include "pycuphy_rsrp.hpp"
#include "pycuphy_ldpc.hpp"
#include "test_pusch_rx.hpp"


namespace pycuphy {

TestPuschRxPipeline::TestPuschRxPipeline(PuschParams& puschParams, const cudaStream_t cudaStream):
m_cudaStream(cudaStream),
m_chEstimator(puschParams, cudaStream),
m_noiseIntfEstimator(cudaStream),
m_chEqualizer(cudaStream),
m_rsrpEstimator(cudaStream),
m_derateMatch(true, cudaStream),
m_decoder(cudaStream),
m_crcChecker(cudaStream)
{}

bool TestPuschRxPipeline::runTest(PuschParams& puschParams, std::string& errMsg) {

    // TODO:
    // - check post-eq SINR (+ other outputs)
    // - add CFO/TA estimation

    // Allocate HARQ buffer.
    void** deRmOutput;
    CUDA_CHECK(cudaMallocHost(&deRmOutput, sizeof(void*) * MAX_N_TBS_SUPPORTED));

    const PerTbParams* pTbPrmsCpu = puschParams.getPerTbPrmsCpuPtr();
    uint32_t NUM_BYTES_PER_LLR = 2;
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        size_t nBytesDeRm = NUM_BYTES_PER_LLR * pTbPrmsCpu[ueIdx].Ncb_padded * pTbPrmsCpu[ueIdx].num_CBs;
        CUDA_CHECK(cudaMalloc(&deRmOutput[ueIdx], nBytesDeRm));
    }

    // Run the pipeline component by component.
    m_chEstimator.estimate(puschParams);
    m_noiseIntfEstimator.estimate(puschParams);
    m_chEqualizer.equalize(puschParams);
    m_rsrpEstimator.estimate(puschParams);

    const std::vector<cuphy::tensor_ref>& llrs = m_chEqualizer.getLlr();
    m_derateMatch.derateMatch(llrs, deRmOutput, puschParams);
    void* decoderOut = m_decoder.decode(deRmOutput, puschParams);
    m_crcChecker.checkCrc(decoderOut, puschParams);

    // Check CRCs.
    const uint32_t* pTbCrcs = m_crcChecker.getTbCrcs();
    uint32_t totNumTbs = m_crcChecker.getTotNumTbs();
    cuphy::tensor_device dTbCrcs = cuphy::tensor_device((void*)pTbCrcs, CUPHY_R_32U, totNumTbs, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned hTbCrcs = cuphy::tensor_pinned(CUPHY_R_32U, totNumTbs, cuphy::tensor_flags::align_tight);
    hTbCrcs.convert(dTbCrcs, m_cudaStream);  // Move to host.

    uint32_t* pTbCrcHost = (uint32_t*)hTbCrcs.addr();
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        if(pTbCrcHost[ueIdx] != 0) {
            errMsg = "CRC failure!";
            return false;
        }
    }

    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        CUDA_CHECK(cudaFree(deRmOutput[ueIdx]));
    }
    CUDA_CHECK(cudaFreeHost(deRmOutput));

    return true;
}



} // namespace pycuphy
