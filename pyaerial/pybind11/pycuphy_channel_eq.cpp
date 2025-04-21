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
#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"
#include "pycuphy_channel_eq.hpp"
#include "tensor_desc.hpp"


namespace py = pybind11;


namespace pycuphy {


ChannelEqualizer::ChannelEqualizer(cudaStream_t cuStream):
m_cuStream(cuStream),
m_linearAlloc(getBufferSize()),
m_kernelStatDescr("ChEqStatDescr"),
m_kernelDynDescr("ChEqDynDescr") {

    m_tCoef.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tReeDiagInv.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDbg.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tDataEq.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tLLR.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tLLRCdm1.resize(MAX_N_USER_GROUPS_SUPPORTED);

    // Allocate descriptors.
    allocateDescr();

    auto statCpuDescrStartAddrs = m_kernelStatDescr.getCpuStartAddrs();
    auto statGpuDescrStartAddrs = m_kernelStatDescr.getGpuStartAddrs();

    bool enableCpuToGpuDescrAsyncCpy = false;
    const uint8_t enableDftSOfdm = 0;  // DFT-S-OFDM not supported by pyAerial.
    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceTime;
    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceFreq;
    const uint8_t enableDebugEqOutput = 1; // Enable debugging output
    cuphyStatus_t status = cuphyCreatePuschRxChEq(m_ctx.handle(),
                                                  &m_chEqHndl,
                                                  tInfoDftBluesteinWorkspaceTime,
                                                  tInfoDftBluesteinWorkspaceFreq,
                                                  800, // GPU device architecture, relevant for DFT-S-OFDM
                                                  enableDftSOfdm,
                                                  enableDebugEqOutput,
                                                  enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                  reinterpret_cast<void**>(&statCpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                                  reinterpret_cast<void**>(&statGpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                                  reinterpret_cast<void**>(&statCpuDescrStartAddrs[PUSCH_CH_EQ_IDFT]),
                                                  reinterpret_cast<void**>(&statGpuDescrStartAddrs[PUSCH_CH_EQ_IDFT]),
                                                  m_cuStream);
    if(CUPHY_STATUS_SUCCESS != status) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEq()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        m_kernelStatDescr.asyncCpuToGpuCpy(m_cuStream);
    }
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));
}


size_t ChannelEqualizer::getBufferSize() const {
    static constexpr uint32_t N_BYTES_C16        = sizeof(data_type_traits<CUPHY_C_16F>::type);
    static constexpr uint32_t N_BYTES_R16        = sizeof(data_type_traits<CUPHY_R_16F>::type);
    static constexpr uint32_t N_BYTES_R32        = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32        = sizeof(data_type_traits<CUPHY_C_32F>::type);

    size_t nBytesBuffer = 0;
    const uint32_t NF = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB;
    const uint32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES;
    const uint32_t N_MAX_LAYERS = MAX_N_ANTENNAS_SUPPORTED;

    // Equalizer coefficients
    uint32_t maxBytesEqualizer = N_BYTES_C32 * MAX_N_ANTENNAS_SUPPORTED * MAX_N_ANTENNAS_SUPPORTED * NF * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST;
    nBytesBuffer += maxBytesEqualizer + EXTRA_PADDING;

    // ReeDiagInv
    uint32_t maxBytesPrecesion = N_BYTES_R32 * MAX_N_ANTENNAS_SUPPORTED * NF * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST;
    nBytesBuffer += maxBytesPrecesion + EXTRA_PADDING;

    // Equalizer debug buffer
    uint32_t maxBytesEqualizerDbg = N_BYTES_C32 * (2 * MAX_N_ANTENNAS_SUPPORTED) * MAX_N_ANTENNAS_SUPPORTED * NF * (OFDM_SYMBOLS_PER_SLOT - 1);
    nBytesBuffer += maxBytesEqualizerDbg + EXTRA_PADDING;

    // Estimated data buffer
    uint32_t maxBytesEstimatedData = N_BYTES_C16 * MAX_N_ANTENNAS_SUPPORTED * (OFDM_SYMBOLS_PER_SLOT - 1);
    nBytesBuffer += maxBytesEstimatedData + EXTRA_PADDING;

    uint32_t maxBitsPerQam = 8;
    uint32_t maxBytesEqOutLLRs = N_BYTES_R16 * NF * maxBitsPerQam * N_MAX_LAYERS * (OFDM_SYMBOLS_PER_SLOT - 1);
    nBytesBuffer += maxBytesEqOutLLRs + EXTRA_PADDING;
    nBytesBuffer += maxBytesEqOutLLRs + EXTRA_PADDING; // For LLR CDM1.

    return nBytesBuffer;
}


void ChannelEqualizer::allocateDescr() {

    std::array<size_t, N_CH_EQ_DESCR_TYPES> statDescrSizeBytes{};
    std::array<size_t, N_CH_EQ_DESCR_TYPES> statDescrAlignBytes{};
    std::array<size_t, N_CH_EQ_DESCR_TYPES> dynDescrSizeBytes{};
    std::array<size_t, N_CH_EQ_DESCR_TYPES> dynDescrAlignBytes{};

    size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();
    size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
    size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();

    // Same stat descriptor is reused for soft-demap as well.
    cuphyStatus_t status = cuphyPuschRxChEqGetDescrInfo(&pStatDescrSizeBytes[PUSCH_CH_EQ_COEF],
                                                        &pStatDescrAlignBytes[PUSCH_CH_EQ_COEF],
                                                        &pStatDescrSizeBytes[PUSCH_CH_EQ_IDFT],
                                                        &pStatDescrAlignBytes[PUSCH_CH_EQ_IDFT],
                                                        &pDynDescrSizeBytes[PUSCH_CH_EQ_COEF],
                                                        &pDynDescrAlignBytes[PUSCH_CH_EQ_COEF],
                                                        &pDynDescrSizeBytes[PUSCH_CH_EQ_SOFT_DEMAP],
                                                        &pDynDescrAlignBytes[PUSCH_CH_EQ_SOFT_DEMAP]);
    if(CUPHY_STATUS_SUCCESS != status) {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxChEqGetDescrInfo()");
    }

    for(uint32_t chEqInstIdx = 1; chEqInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ; ++chEqInstIdx) {
        pStatDescrSizeBytes[PUSCH_CH_EQ_COEF + chEqInstIdx]  = pStatDescrSizeBytes[PUSCH_CH_EQ_COEF];
        pStatDescrAlignBytes[PUSCH_CH_EQ_COEF + chEqInstIdx] = pStatDescrAlignBytes[PUSCH_CH_EQ_COEF];
        pDynDescrSizeBytes[PUSCH_CH_EQ_COEF + chEqInstIdx]   = pDynDescrSizeBytes[PUSCH_CH_EQ_COEF];
        pDynDescrAlignBytes[PUSCH_CH_EQ_COEF + chEqInstIdx]  = pDynDescrAlignBytes[PUSCH_CH_EQ_COEF];
    }

    // Allocate descriptors (CPU and GPU).
    m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes);
    m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes);
}


void ChannelEqualizer::equalize(PuschParams& puschParams) {

    m_linearAlloc.reset();

    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    std::vector<cuphy::tensor_device>& tDataRx = puschParams.getDataTensor();

    uint16_t nUeGrps = puschParams.getNumUeGrps();

    // Allocate output tensor arrays in device memory.
    void* initAddr = static_cast<char*>(m_linearAlloc.address());
    for(int i = 0; i < nUeGrps; ++i) {
        int nDataSym = pPuschRxUeGrpPrmsCpu[i].nDataSym;
        int nDmrsSym = pPuschRxUeGrpPrmsCpu[i].nDmrsSyms;
        int numCh = pPuschRxUeGrpPrmsCpu[i].dmrsAddlnPos + 1;

        uint8_t nDmrsCdmGrpsNoData = pPuschRxUeGrpPrmsCpu[i].nDmrsCdmGrpsNoData;
        if(nDmrsCdmGrpsNoData == 1) {
            nDataSym += nDmrsSym;
        }

        uint16_t cellPrmDynIdx = puschParams.m_puschDynPrms.pCellGrpDynPrm->pUeGrpPrms[i].pCellPrm->cellPrmDynIdx;
        copyTensorData(tDataRx[cellPrmDynIdx], pPuschRxUeGrpPrmsCpu[i].tInfoDataRx);

        m_tCoef[i].desc().set(CUPHY_C_32F,
                              pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                              CUPHY_N_TONES_PER_PRB,
                              pPuschRxUeGrpPrmsCpu[i].nLayers,
                              pPuschRxUeGrpPrmsCpu[i].nPrb,
                              numCh,
                              cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tCoef[i]);
        copyTensorData(m_tCoef[i], pPuschRxUeGrpPrmsCpu[i].tInfoEqCoef);

        m_tReeDiagInv[i].desc().set(CUPHY_R_32F,
                                    CUPHY_N_TONES_PER_PRB,
                                    pPuschRxUeGrpPrmsCpu[i].nLayers,
                                    pPuschRxUeGrpPrmsCpu[i].nPrb,
                                    pPuschRxUeGrpPrmsCpu[i].nTimeChEsts,
                                    cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tReeDiagInv[i]);
        copyTensorData(m_tReeDiagInv[i], pPuschRxUeGrpPrmsCpu[i].tInfoReeDiagInv);

        m_tDbg[i].desc().set(CUPHY_C_32F,
                             pPuschRxUeGrpPrmsCpu[i].nLayers + pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                             pPuschRxUeGrpPrmsCpu[i].nLayers,
                             CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                             numCh,
                             cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDbg[i]);
        copyTensorData(m_tDbg[i], pPuschRxUeGrpPrmsCpu[i].tInfoChEqDbg);

        m_tDataEq[i].desc().set(CUPHY_C_16F,
                                pPuschRxUeGrpPrmsCpu[i].nLayers,
                                CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                                nDataSym,
                                cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tDataEq[i]);
        copyTensorData(m_tDataEq[i], pPuschRxUeGrpPrmsCpu[i].tInfoDataEq);

        m_tLLR[i].desc().set(CUPHY_R_16F,
                             CUPHY_QAM_256,
                             pPuschRxUeGrpPrmsCpu[i].nLayers,
                             CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                             nDataSym,
                             cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tLLR[i]);
        copyTensorData(m_tLLR[i], pPuschRxUeGrpPrmsCpu[i].tInfoLLR);

        m_tLLRCdm1[i].desc().set(CUPHY_R_16F,
                                 CUPHY_QAM_256,
                                 pPuschRxUeGrpPrmsCpu[i].nLayers,
                                 CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                                 nDataSym,
                                 cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tLLRCdm1[i]);
        copyTensorData(m_tLLRCdm1[i], pPuschRxUeGrpPrmsCpu[i].tInfoLLRCdm1);
    }

    size_t finalOffset = m_linearAlloc.offset();
    CUDA_CHECK(cudaMemsetAsync(initAddr, 0, finalOffset, m_cuStream));

    cuphyPuschRxChEqLaunchCfgs_t chEqCoefCompLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];
    for(int32_t chEqTimeInst = 0; chEqTimeInst < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEqTimeInst) {
        chEqCoefCompLaunchCfgs[chEqTimeInst].nCfgs = 0;
    }
    cuphyPuschRxChEqLaunchCfgs_t chEqSoftDemapLaunchCfgs;
    chEqSoftDemapLaunchCfgs.nCfgs = 0;

    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t coefComputeSetupStatus = cuphySetupPuschRxChEqCoefCompute(m_chEqHndl,
                                                                            pPuschRxUeGrpPrmsCpu,
                                                                            pPuschRxUeGrpPrmsGpu,
                                                                            nUeGrps,
                                                                            puschParams.getMaxNumPrb(),
                                                                            puschParams.m_puschStatPrms.enableCfoCorrection,
                                                                            puschParams.m_puschStatPrms.enablePuschTdi,
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            reinterpret_cast<void**>(&dynCpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                                                            reinterpret_cast<void**>(&dynGpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                                                            chEqCoefCompLaunchCfgs,
                                                                            m_cuStream);

    const uint16_t symbolBitMask = CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK;
    cuphyStatus_t setupSoftDemapStatus = cuphySetupPuschRxChEqSoftDemap(m_chEqHndl,
                                                                        pPuschRxUeGrpPrmsCpu,
                                                                        pPuschRxUeGrpPrmsGpu,
                                                                        nUeGrps,
                                                                        puschParams.getMaxNumPrb(),
                                                                        puschParams.m_puschStatPrms.enableCfoCorrection,
                                                                        puschParams.m_puschStatPrms.enablePuschTdi,
                                                                        symbolBitMask,
                                                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                        static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                        static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                        &chEqSoftDemapLaunchCfgs,
                                                                        m_cuStream);

    if(CUPHY_STATUS_SUCCESS != coefComputeSetupStatus) {
        throw cuphy::cuphy_fn_exception(coefComputeSetupStatus, "cuphySetupPuschRxChEqCoefCompute()");
    }

    if(CUPHY_STATUS_SUCCESS != setupSoftDemapStatus) {
        throw cuphy::cuphy_fn_exception(setupSoftDemapStatus, "cuphySetupPuschRxChEqSoftDemap()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        m_kernelDynDescr.asyncCpuToGpuCpy(m_cuStream);
        puschParams.copyPuschRxUeGrpPrms();
    }

    for(uint32_t chEqInstIdx = 0; chEqInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ; ++chEqInstIdx) {
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < chEqCoefCompLaunchCfgs[chEqInstIdx].nCfgs; ++hetCfgIdx) {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = chEqCoefCompLaunchCfgs[chEqInstIdx].cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                                             kernelNodeParamsDriver.gridDimX,
                                                             kernelNodeParamsDriver.gridDimY,
                                                             kernelNodeParamsDriver.gridDimZ,
                                                             kernelNodeParamsDriver.blockDimX,
                                                             kernelNodeParamsDriver.blockDimY,
                                                             kernelNodeParamsDriver.blockDimZ,
                                                             kernelNodeParamsDriver.sharedMemBytes,
                                                             static_cast<CUstream>(m_cuStream),
                                                             kernelNodeParamsDriver.kernelParams,
                                                             kernelNodeParamsDriver.extra));
        }
    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < chEqSoftDemapLaunchCfgs.nCfgs; ++hetCfgIdx) {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = chEqSoftDemapLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                                         kernelNodeParamsDriver.gridDimX,
                                                         kernelNodeParamsDriver.gridDimY,
                                                         kernelNodeParamsDriver.gridDimZ,
                                                         kernelNodeParamsDriver.blockDimX,
                                                         kernelNodeParamsDriver.blockDimY,
                                                         kernelNodeParamsDriver.blockDimZ,
                                                         kernelNodeParamsDriver.sharedMemBytes,
                                                         static_cast<CUstream>(m_cuStream),
                                                         kernelNodeParamsDriver.kernelParams,
                                                         kernelNodeParamsDriver.extra));
    }

}


ChannelEqualizer::~ChannelEqualizer() {
    destroy();
}


void ChannelEqualizer::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxChEq(m_chEqHndl);
    if(CUPHY_STATUS_SUCCESS != status) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxChEq()");
    }
}


PyChannelEqualizer::PyChannelEqualizer(uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_chEqualizer((cudaStream_t)cuStream) {
    m_tChannelEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tInfoLwInv.resize(MAX_N_USER_GROUPS_SUPPORTED);
}


const std::vector<py::array_t<float>>&  PyChannelEqualizer::equalize(const std::vector<py::array_t<std::complex<float>>>& chEst,
                                                                     const std::vector<py::array_t<std::complex<float>>>& infoLwInv,
                                                                     const py::array& infoNoiseVarPreEq,
                                                                     const py::array& invNoiseVarLin,
                                                                     PuschParams& puschParams) {
    m_LLR.clear();
    m_coef.clear();
    m_ReeDiagInv.clear();
    m_dataEq.clear();

    m_tInfoNoiseVarPreEq = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
        infoNoiseVarPreEq,
        CUPHY_R_32F,
        CUPHY_R_32F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);

    py::buffer_info buf = invNoiseVarLin.request();
    float* invNoiseVarLinPtr = (float*)buf.ptr;

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    for(int i = 0; i < nUeGrps; ++i) {
        int nDataSym = pPuschRxUeGrpPrmsCpu[i].nDataSym;
        int nDmrsSym = pPuschRxUeGrpPrmsCpu[i].nDmrsSyms;
        int numCh = pPuschRxUeGrpPrmsCpu[i].dmrsAddlnPos + 1;

        m_tChannelEst[i] = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
            chEst[i],
            CUPHY_C_32F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        copyTensorData(m_tChannelEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoHEst);

        m_tInfoLwInv[i] = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
            infoLwInv[i],
            CUPHY_C_32F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        copyTensorData(m_tInfoLwInv[i], pPuschRxUeGrpPrmsCpu[i].tInfoLwInv);

        copyTensorData(m_tInfoNoiseVarPreEq, pPuschRxUeGrpPrmsCpu[i].tInfoNoiseVarPreEq);

        pPuschRxUeGrpPrmsCpu[i].invNoiseVarLin = invNoiseVarLinPtr[i];
    }

    m_chEqualizer.equalize(puschParams);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Fetch outputs.
    const std::vector<cuphy::tensor_ref>& dLlr = m_chEqualizer.getLlr();
    const std::vector<cuphy::tensor_ref>& dCoef = m_chEqualizer.getChEqCoef();
    const std::vector<cuphy::tensor_ref>& dReeDiagInv = m_chEqualizer.getReeDiagInv();
    const std::vector<cuphy::tensor_ref>& dDataEq = m_chEqualizer.getDataEq();

    // Outputs.
    for(int i = 0; i < nUeGrps; ++i) {

        int nDataSym = pPuschRxUeGrpPrmsCpu[i].nDataSym;
        int nDmrsSym = pPuschRxUeGrpPrmsCpu[i].nDmrsSyms;
        uint8_t nDmrsCdmGrpsNoData = pPuschRxUeGrpPrmsCpu[i].nDmrsCdmGrpsNoData;
        if(nDmrsCdmGrpsNoData == 1) {
            nDataSym += nDmrsSym;
        }

        cuphy::tensor_pinned llr = cuphy::tensor_pinned(CUPHY_R_32F,
                                                        CUPHY_QAM_256,
                                                        pPuschRxUeGrpPrmsCpu[i].nLayers,
                                                        CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                                                        nDataSym,
                                                        cuphy::tensor_flags::align_tight);
        llr.convert(dLlr[i], m_cuStream);

        cuphy::tensor_pinned coef = cuphy::tensor_pinned(CUPHY_C_32F,
                                                         pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                                                         CUPHY_N_TONES_PER_PRB,
                                                         pPuschRxUeGrpPrmsCpu[i].nLayers,
                                                         pPuschRxUeGrpPrmsCpu[i].nPrb,
                                                         pPuschRxUeGrpPrmsCpu[i].nTimeChEsts,
                                                         cuphy::tensor_flags::align_tight);
        coef.convert(dCoef[i], m_cuStream);

        cuphy::tensor_pinned reeDiagInv = cuphy::tensor_pinned(CUPHY_R_32F,
                                                               CUPHY_N_TONES_PER_PRB,
                                                               pPuschRxUeGrpPrmsCpu[i].nLayers,
                                                               pPuschRxUeGrpPrmsCpu[i].nPrb,
                                                               pPuschRxUeGrpPrmsCpu[i].nTimeChEsts,
                                                               cuphy::tensor_flags::align_tight);
        reeDiagInv.convert(dReeDiagInv[i], m_cuStream);


        cuphy::tensor_pinned dataEq = cuphy::tensor_pinned(CUPHY_C_32F,
                                                           pPuschRxUeGrpPrmsCpu[i].nLayers,
                                                           CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                                                           nDataSym,
                                                           cuphy::tensor_flags::align_tight);
        dataEq.convert(dDataEq[i], m_cuStream);

        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

        m_LLR.push_back(
            hostToNumpy<float>(
                (float*)llr.addr(),
                CUPHY_QAM_256,
                pPuschRxUeGrpPrmsCpu[i].nLayers,
                CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                nDataSym)
        );

        m_coef.push_back(
            hostToNumpy<std::complex<float>>(
                (std::complex<float>*)coef.addr(),
                pPuschRxUeGrpPrmsCpu[i].nRxAnt,
                CUPHY_N_TONES_PER_PRB,
                pPuschRxUeGrpPrmsCpu[i].nLayers,
                pPuschRxUeGrpPrmsCpu[i].nPrb,
                pPuschRxUeGrpPrmsCpu[i].nTimeChEsts)
        );

        m_ReeDiagInv.push_back(
            hostToNumpy<float>(
                (float*)reeDiagInv.addr(),
                CUPHY_N_TONES_PER_PRB,
                pPuschRxUeGrpPrmsCpu[i].nLayers,
                pPuschRxUeGrpPrmsCpu[i].nPrb,
                pPuschRxUeGrpPrmsCpu[i].nTimeChEsts)
        );

        m_dataEq.push_back(
            hostToNumpy<std::complex<float>>(
                (std::complex<float>*)dataEq.addr(),
                pPuschRxUeGrpPrmsCpu[i].nLayers,
                CUPHY_N_TONES_PER_PRB * pPuschRxUeGrpPrmsCpu[i].nPrb,
                nDataSym)
        );

    }

    return m_LLR;
}

}
