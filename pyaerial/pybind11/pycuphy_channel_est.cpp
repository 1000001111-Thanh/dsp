/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_debug.hpp"
#include "pycuphy_channel_est.hpp"
#include "tensor_desc.hpp"

#include "cuphy_factory.hpp"

namespace py = pybind11;

namespace pycuphy {


void ChannelEstimator::allocateDescr() {

    std::array<size_t, N_CH_EST_DESCR_TYPES> statDescrSizeBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> statDescrAlignBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> dynDescrSizeBytes{};
    std::array<size_t, N_CH_EST_DESCR_TYPES> dynDescrAlignBytes{};

    size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();
    size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
    size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();

    cuphyStatus_t status = cuphyPuschRxChEstGetDescrInfo(&pStatDescrSizeBytes[CH_EST],
                                                         &pStatDescrAlignBytes[CH_EST],
                                                         &pDynDescrSizeBytes[CH_EST],
                                                         &pDynDescrAlignBytes[CH_EST]);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxChEstGetDescrInfo()");
    }

    for(uint32_t chEstTimeIdx = 1; chEstTimeIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeIdx) {
        pStatDescrSizeBytes[CH_EST + chEstTimeIdx]  = pStatDescrSizeBytes[CH_EST];
        pStatDescrAlignBytes[CH_EST + chEstTimeIdx] = pStatDescrAlignBytes[CH_EST];
        pDynDescrSizeBytes[CH_EST + chEstTimeIdx]   = pDynDescrSizeBytes[CH_EST];
        pDynDescrAlignBytes[CH_EST + chEstTimeIdx]  = pDynDescrAlignBytes[CH_EST];
    }

    // Allocate descriptors (CPU and GPU).
    m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes);
    m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes);
}


ChannelEstimator::ChannelEstimator(const PuschParams& puschParams, cudaStream_t cuStream):
m_linearAlloc(getBufferSize()),
m_kernelStatDescr("ChEstStatDescr"),
m_kernelDynDescr("ChEstDynDescr"),
m_cuStream(cuStream) {
    init(puschParams);
}


ChannelEstimator::~ChannelEstimator() {
    destroy();
}


void ChannelEstimator::debugDump(H5DebugDump& debugDump, uint16_t numUeGrps, cudaStream_t cuStream) {
    for(int i = 0; i < numUeGrps; i++) {
        debugDump.dump(std::string("LsChEst" + std::to_string(i)), m_tDmrsLSEst[i], cuStream);
        debugDump.dump(std::string("ChEst" + std::to_string(i)), m_tChannelEst[i], cuStream);
    }
}


size_t ChannelEstimator::getBufferSize() const {
    static constexpr uint32_t N_BYTES_R32 = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32 = sizeof(data_type_traits<CUPHY_C_32F>::type);

    size_t nBytesBuffer = 0;
    const uint32_t NF = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB;
    const uint32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES;
    const uint32_t MAX_NUM_DMRS_LAYERS = 8;

    nBytesBuffer += N_BYTES_C32 * MAX_N_ANTENNAS_SUPPORTED * MAX_N_ANTENNAS_SUPPORTED * NF * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST + EXTRA_PADDING;
    // m_tDmrsDelayMean
    nBytesBuffer += N_BYTES_R32 * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST + EXTRA_PADDING;
    // m_tDmrsAccum
    nBytesBuffer += N_BYTES_C32 * 2 + EXTRA_PADDING;
    // m_tDmrsLSEst
    nBytesBuffer += N_BYTES_C32 * (NF / 2) * MAX_NUM_DMRS_LAYERS * MAX_N_ANTENNAS_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST  + EXTRA_PADDING;

    // Debug buffer
    uint32_t maxBytesChEstDbg = N_BYTES_C32 * (NF / 2) * MAX_N_DMRSSYMS_SUPPORTED;
    nBytesBuffer += CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST * maxBytesChEstDbg + EXTRA_PADDING;

    return nBytesBuffer;
}


void ChannelEstimator::init(const PuschParams& puschParams) {

    // Channel estimation algorithm. RKHS not yet supported by pyAerial.
    cuphyPuschChEstAlgoType_t algoType = puschParams.m_puschStatPrms.chEstAlgo;
    if(algoType == PUSCH_CH_EST_ALGO_TYPE_RKHS) {
        throw std::invalid_argument("RKHS not supported by pyAerial yet.");
    }

    // Same as in pusch_utils.hpp
    const auto nMaxChEstHetCfgs = (algoType == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
        CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS : CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS;

    m_tChannelEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDbg.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDmrsLSEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDmrsDelayMean.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDmrsAccum.resize(MAX_N_USER_GROUPS_SUPPORTED);

    // Allocate descriptors.
    allocateDescr();

    // Create the channel estimator object.
    auto statCpuDescrStartAddrs = m_kernelStatDescr.getCpuStartAddrs();
    auto statGpuDescrStartAddrs = m_kernelStatDescr.getGpuStartAddrs();
    bool enableCpuToGpuDescrAsyncCpy = false;
    m_chestKernelBuilder = cuphy::factory::createPuschRxChEstKernelBuilder();
    auto [puschRxChEst, status] = cuphy::factory::createPuschRxChEst(m_chestKernelBuilder.get(),
                                                   puschParams.m_puschStatPrms.pWFreq,
                                                   puschParams.m_puschStatPrms.pWFreq4,
                                                   puschParams.m_puschStatPrms.pWFreqSmall,
                                                   puschParams.m_puschStatPrms.pShiftSeq,
                                                   puschParams.m_puschStatPrms.pShiftSeq4,
                                                   puschParams.m_puschStatPrms.pUnShiftSeq,
                                                   puschParams.m_puschStatPrms.pUnShiftSeq4,
                                                   nMaxChEstHetCfgs,
                                                   puschParams.m_puschStatPrms.enableEarlyHarq,
                                                   algoType,
                                                   nullptr,
                                                   nullptr, // RKHS paramaters, disabled for now
                                                   enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                   reinterpret_cast<void**>(&statCpuDescrStartAddrs[CH_EST]),
                                                   reinterpret_cast<void**>(&statGpuDescrStartAddrs[CH_EST]),
                                                   m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEst()");
    }

    m_chest = std::move(puschRxChEst);

    if(!enableCpuToGpuDescrAsyncCpy){
        m_kernelStatDescr.asyncCpuToGpuCpy(m_cuStream);
    }
}


void ChannelEstimator::estimate(PuschParams& puschParams) {

    m_linearAlloc.reset();

    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    std::vector<cuphy::tensor_device>& tDataRx = puschParams.getDataTensor();
    uint16_t nUeGrps = puschParams.getNumUeGrps();

    // Allocate output tensor arrays in device memory.
    for(int i = 0; i < nUeGrps; ++i) {
        int numCh = pPuschRxUeGrpPrmsCpu[i].dmrsAddlnPos + 1;

        uint16_t cellPrmDynIdx = puschParams.m_puschDynPrms.pCellGrpDynPrm->pUeGrpPrms[i].pCellPrm->cellPrmDynIdx;
        copyTensorData(tDataRx[cellPrmDynIdx], pPuschRxUeGrpPrmsCpu[i].tInfoDataRx);

        m_tChannelEst[i].desc().set(CUPHY_C_32F,
                                    pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                    pPuschRxUeGrpPrmsCpu[i].nLayers,
                                    CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                                    numCh,
                                    cuphy::tensor_flags::align_default);
        m_linearAlloc.alloc(m_tChannelEst[i]);
        copyTensorData(m_tChannelEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoHEst);

        m_tDbg[i].desc().set(CUPHY_C_32F,
                             CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb / 2,
                             pPuschRxUeGrpPrmsCpu[i].nDmrsSyms,
                             1,
                             1,
                             cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDbg[i]);
        copyTensorData(m_tDbg[i], pPuschRxUeGrpPrmsCpu[i].tInfoChEstDbg);

        m_tDmrsDelayMean[i].desc().set(CUPHY_R_32F,
                                       numCh,
                                       cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDmrsDelayMean[i]);
        copyTensorData(m_tDmrsDelayMean[i], pPuschRxUeGrpPrmsCpu[i].tInfoDmrsDelayMean);

        m_tDmrsLSEst[i].desc().set(CUPHY_C_32F,
                                   CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb / 2,
                                   pPuschRxUeGrpPrmsCpu[i].nLayers,
                                   pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                   numCh,
                                   cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDmrsLSEst[i]);
        copyTensorData(m_tDmrsLSEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoDmrsLSEst);

        m_tDmrsAccum[i].desc().set(CUPHY_C_32F, 2, cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDmrsAccum[i]);
        copyTensorData(m_tDmrsAccum[i], pPuschRxUeGrpPrmsCpu[i].tInfoDmrsAccum);
    }
    m_linearAlloc.memset(0.f, m_cuStream);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Run setup.
    const uint8_t enableDftSOfdm = 0;
    const uint8_t chEstAlgo = static_cast<uint8_t>(puschParams.m_puschStatPrms.chEstAlgo);
    const uint8_t enablePerPrgChEst = puschParams.m_puschStatPrms.enablePerPrgChEst;
    const uint16_t waitTimeOutPreEarlyHarqUs = 0;
    const uint16_t waitTimeOutPostEarlyHarqUs = 0;
    const uint8_t enableEarlyHarqProc = 0;
    const uint8_t enableFrontLoadedDmrsProc = 0;
    const uint8_t enableDeviceGraphLaunch = 0;
    bool enableCpuToGpuDescrAsyncCpy = false;
    uint8_t preEarlyHarqWaitKernelStatus_d = 0;
    uint8_t postEarlyHarqWaitKernelStatus_d = 0;
    cuphyStatus_t chEstSetupStatus = m_chest->setup(m_chestKernelBuilder.get(),
                                                            pPuschRxUeGrpPrmsCpu,
                                                            pPuschRxUeGrpPrmsGpu,
                                                            nUeGrps,
                                                            enableDftSOfdm,
                                                            chEstAlgo,
                                                            enablePerPrgChEst,
                                                            &preEarlyHarqWaitKernelStatus_d,
                                                            &postEarlyHarqWaitKernelStatus_d,
                                                            waitTimeOutPreEarlyHarqUs,
                                                            waitTimeOutPostEarlyHarqUs,
                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                            reinterpret_cast<void**>(&dynCpuDescrStartAddrs[CH_EST]),
                                                            reinterpret_cast<void**>(&dynGpuDescrStartAddrs[CH_EST]),
                                                            enableEarlyHarqProc,
                                                            enableFrontLoadedDmrsProc,
                                                            enableDeviceGraphLaunch,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            nullptr,
                                                            m_cuStream);
    if(chEstSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(chEstSetupStatus, "cuphySetupPuschRxChEst()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        m_kernelDynDescr.asyncCpuToGpuCpy(m_cuStream);
        puschParams.copyPuschRxUeGrpPrms();
    }

    // Launch kernel using the CUDA driver API.
    m_chest->runKernels(m_cuStream);
    if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
        m_chest->runSecondaryKernels(m_cuStream);
    }
}


void ChannelEstimator::destroy() {
    // Destroy the PUSCH channel estimation handle.
    m_chest.reset();
    m_chestKernelBuilder.reset();
}



PyChannelEstimator::PyChannelEstimator(const PuschParams& puschParams, uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_chEstimator(puschParams, (cudaStream_t)cuStream) {}


const std::vector<py::array_t<std::complex<float>>>& PyChannelEstimator::estimate(PuschParams& puschParams) {

    m_chEst.clear();

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();

    // Channel estimation algorithm. RKHS not yet supported by pyAerial.
    cuphyPuschChEstAlgoType_t chEstAlgo = puschParams.m_puschStatPrms.chEstAlgo;
    if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS) {
        throw std::invalid_argument("RKHS not supported by pyAerial yet.");
    }

    // Run channel estimation.
    m_chEstimator.estimate(puschParams);

    // Create the return values to Python.
    if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE || chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
        const std::vector<cuphy::tensor_ref>& channelEst = m_chEstimator.getChEst();

        for(int i = 0; i < nUeGrps; ++i) {

            uint32_t dim0 = pPuschRxUeGrpPrmsCpu[i].nRxAnt;
            uint32_t dim1 = pPuschRxUeGrpPrmsCpu[i].nLayers;
            uint32_t dim2 = CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb;
            uint32_t dim3 = pPuschRxUeGrpPrmsCpu[i].nTimeChEsts;


            cuphy::tensor_pinned hostChannelEst = cuphy::tensor_pinned(CUPHY_C_32F,
                                                                       dim0,
                                                                       dim1,
                                                                       dim2,
                                                                       dim3,
                                                                       cuphy::tensor_flags::align_tight);
            hostChannelEst.convert(channelEst[i], m_cuStream);
            CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

            m_chEst.push_back(
                hostToNumpy<std::complex<float>>((std::complex<float>*)hostChannelEst.addr(), dim0, dim1, dim2, dim3)
            );
        }
    }
    else if (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_LS_ONLY) {

        const std::vector<cuphy::tensor_ref>& lsChEst = m_chEstimator.getLsChEst();

        for(int i = 0; i < nUeGrps; ++i) {

            uint32_t dim0 = CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb / 2;
            uint32_t dim1 = pPuschRxUeGrpPrmsCpu[i].nLayers;
            uint32_t dim2 = pPuschRxUeGrpPrmsCpu[i].nRxAnt;
            uint32_t dim3 = pPuschRxUeGrpPrmsCpu[i].nTimeChEsts;

            cuphy::tensor_pinned hostLsEst = cuphy::tensor_pinned(CUPHY_C_32F,
                                                                  dim0,
                                                                  dim1,
                                                                  dim2,
                                                                  dim3,
                                                                  cuphy::tensor_flags::align_tight);
            hostLsEst.convert(lsChEst[i], m_cuStream);
            CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

            m_chEst.push_back(
                hostToNumpy<std::complex<float>>((std::complex<float>*)hostLsEst.addr(), dim0, dim1, dim2, dim3)
            );
        }
    }
    return m_chEst;
}


} // namespace pycuphy
