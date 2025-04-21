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
#include "pycuphy_cfo_ta_est.hpp"
#include "tensor_desc.hpp"


namespace py = pybind11;


namespace pycuphy {

CfoTaEstimator::~CfoTaEstimator() {
    destroy();
}


CfoTaEstimator::CfoTaEstimator(cudaStream_t cuStream):
m_linearAlloc(getBufferSize()),
m_cuStream(cuStream) {

    // Allocate descriptors.
    allocateDescr();

    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t status = cuphyCreatePuschRxCfoTaEst(&m_cfoTaEstHndl,
                                                      enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                      static_cast<void*>(m_statDescrBufCpu.addr()),
                                                      static_cast<void*>(m_statDescrBufGpu.addr()),
                                                      m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxCfoTaEst()");
    }
}


size_t CfoTaEstimator::getBufferSize() const {

    static constexpr uint32_t N_BYTES_R32 = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32 = sizeof(data_type_traits<CUPHY_C_32F>::type);
    static constexpr uint32_t N_BYTES_PER_UINT32 = 4;
    const int32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES;
    static constexpr uint32_t MAX_N_UE = MAX_N_TBS_SUPPORTED;

    size_t nBytesBuffer = 0;

    uint32_t maxBytesCfoEst = N_BYTES_C32 * MAX_ND_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoEst + EXTRA_PADDING;

    nBytesBuffer += N_BYTES_R32 * MAX_N_UE;  // CFO Hz
    nBytesBuffer += N_BYTES_R32 * MAX_N_UE;  // TA

    uint32_t maxBytesCfoPhaseRot = N_BYTES_C32 * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoPhaseRot + EXTRA_PADDING;

    uint32_t maxBytesTaPhaseRot = N_BYTES_C32 * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesTaPhaseRot + EXTRA_PADDING;

    uint32_t maxBytesCfoTaEstInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoTaEstInterCtaSyncCnt + EXTRA_PADDING;

    return nBytesBuffer;
}


void CfoTaEstimator::allocateDescr() {

    size_t statDescrAlignBytes, dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxCfoTaEstGetDescrInfo(&m_statDescrSizeBytes,
                                                                            &statDescrAlignBytes,
                                                                            &m_dynDescrSizeBytes,
                                                                            &dynDescrAlignBytes);
    if(statusGetWorkspaceSize != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusGetWorkspaceSize, "cuphyPuschRxCfoTaEstGetDescrInfo()");
    }

    m_dynDescrSizeBytes = ((m_dynDescrSizeBytes + (dynDescrAlignBytes - 1)) / dynDescrAlignBytes) * dynDescrAlignBytes;
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);

    m_statDescrSizeBytes = ((m_statDescrSizeBytes + (statDescrAlignBytes - 1)) / statDescrAlignBytes) * statDescrAlignBytes;
    m_statDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_statDescrSizeBytes);
    m_statDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_statDescrSizeBytes);
}


void CfoTaEstimator::estimate(PuschParams& puschParams) {

    m_linearAlloc.reset();

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;

    m_tCfoEstVec.resize(nUeGrps);

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    // Allocate output tensor arrays in device memory.
    void* initAddr = static_cast<char*>(m_linearAlloc.address());

    m_tCfoPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tCfoPhaseRot);

    m_tTaPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tTaPhaseRot);

    m_tCfoHz.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tCfoHz);

    m_tTaEst.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tTaEst);

    m_tCfoTaEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, nUeGrps, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tCfoTaEstInterCtaSyncCnt);

    for(int i = 0; i < nUeGrps; ++i) {
        copyTensorData(m_tCfoPhaseRot, pPuschRxUeGrpPrmsCpu[i].tInfoCfoPhaseRot);
        copyTensorData(m_tTaPhaseRot, pPuschRxUeGrpPrmsCpu[i].tInfoTaPhaseRot);
        copyTensorData(m_tCfoHz, pPuschRxUeGrpPrmsCpu[i].tInfoCfoHz);
        copyTensorData(m_tTaEst, pPuschRxUeGrpPrmsCpu[i].tInfoTaEst);
        copyTensorData(m_tCfoTaEstInterCtaSyncCnt, pPuschRxUeGrpPrmsCpu[i].tInfoCfoTaEstInterCtaSyncCnt);

        m_tCfoEstVec[i].desc().set(CUPHY_C_32F, MAX_ND_SUPPORTED, nUes, cuphy::tensor_flags::align_tight);
        m_linearAlloc.alloc(m_tCfoEstVec[i]);
        copyTensorData(m_tCfoEstVec[i], pPuschRxUeGrpPrmsCpu[i].tInfoCfoEst);
    }

    size_t finalOffset = m_linearAlloc.offset();
    CUDA_CHECK(cudaMemsetAsync(initAddr, 0, finalOffset, m_cuStream));

    // Run setup.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyPuschRxCfoTaEstLaunchCfgs_t m_cfoTaEstLaunchCfgs;
    m_cfoTaEstLaunchCfgs.nCfgs = 0;  // Setup within the component.
    cuphyStatus_t cfoTaEstSetupStatus = cuphySetupPuschRxCfoTaEst(m_cfoTaEstHndl,
                                                                  pPuschRxUeGrpPrmsCpu,
                                                                  pPuschRxUeGrpPrmsGpu,
                                                                  nUeGrps,
                                                                  puschParams.getMaxNumPrb(),
                                                                  0,  // pDbg
                                                                  enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                  static_cast<void*>(m_dynDescrBufCpu.addr()),
                                                                  static_cast<void*>(m_dynDescrBufGpu.addr()),
                                                                  &m_cfoTaEstLaunchCfgs,
                                                                  m_cuStream);
    if(cfoTaEstSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(cfoTaEstSetupStatus, "cuphySetupPuschRxCfoTaEst()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(),
                                   m_dynDescrBufCpu.addr(),
                                   m_dynDescrSizeBytes,
                                   cudaMemcpyHostToDevice,
                                   m_cuStream));
        puschParams.copyPuschRxUeGrpPrms();
    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_cfoTaEstLaunchCfgs.nCfgs; ++hetCfgIdx) {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_cfoTaEstLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
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


void CfoTaEstimator::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxCfoTaEst(m_cfoTaEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxCfoTaEst()");
    }
}


PyCfoTaEstimator::PyCfoTaEstimator(uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_cfoTaEstimator((cudaStream_t)cuStream) {
    m_tChannelEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
}


const std::vector<py::array_t<std::complex<float>>>& PyCfoTaEstimator::estimate(const std::vector<py::array_t<std::complex<float>>>& chEst,
                                                                                PuschParams& puschParams) {

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    m_cfoEstVec.clear();

    // Read inputs.
    for(int i = 0; i < nUeGrps; ++i) {
        m_tChannelEst[i] = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
            chEst[i],
            CUPHY_C_32F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        copyTensorData(m_tChannelEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoHEst);
    }

    // Run the estimator.
    m_cfoTaEstimator.estimate(puschParams);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Get the return values.
    const std::vector<cuphy::tensor_ref>& dCfoEst = m_cfoTaEstimator.getCfoEst();
    const cuphy::tensor_ref& dCfoHz = m_cfoTaEstimator.getCfoHz();
    const cuphy::tensor_ref& dTaEst = m_cfoTaEstimator.getTaEst();
    const cuphy::tensor_ref& dCfoPhaseRot = m_cfoTaEstimator.getCfoPhaseRot();
    const cuphy::tensor_ref& dTaPhaseRot = m_cfoTaEstimator.getTaPhaseRot();


    // Move the return values to host and Python.

    cuphy::tensor_pinned cfoHz = cuphy::tensor_pinned(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    cfoHz.convert(dCfoHz, m_cuStream);

    cuphy::tensor_pinned taEst = cuphy::tensor_pinned(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    taEst.convert(dTaEst, m_cuStream);

    cuphy::tensor_pinned cfoPhaseRot = cuphy::tensor_pinned(CUPHY_C_32F,
                                                            CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST,
                                                            CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP,
                                                            nUeGrps,
                                                            cuphy::tensor_flags::align_tight);
    cfoPhaseRot.convert(dCfoPhaseRot, m_cuStream);

    cuphy::tensor_pinned taPhaseRot = cuphy::tensor_pinned(CUPHY_C_32F,
                                                           CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP,
                                                           nUeGrps,
                                                           cuphy::tensor_flags::align_tight);
    taPhaseRot.convert(dTaPhaseRot, m_cuStream);

    for(int i = 0; i < nUeGrps; ++i) {

        cuphy::tensor_pinned cfoEstVec = cuphy::tensor_pinned(CUPHY_C_32F,
                                                              MAX_ND_SUPPORTED,
                                                              pPuschRxUeGrpPrmsCpu[i].nUes,
                                                              cuphy::tensor_flags::align_tight);
        cfoEstVec.convert(dCfoEst[i], m_cuStream);

        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

        m_cfoEstVec.push_back(
            hostToNumpy<std::complex<float>>(
                (std::complex<float>*)cfoEstVec.addr(),
                MAX_ND_SUPPORTED,
                pPuschRxUeGrpPrmsCpu[i].nUes)
        );
    }
    m_cfoEstHz = hostToNumpy<float>((float*)cfoHz.addr(), nUes);
    m_taEst = hostToNumpy<float>((float*)taEst.addr(), nUes);
    m_cfoPhaseRot = hostToNumpy<std::complex<float>>((std::complex<float>*)cfoPhaseRot.addr(), CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps);
    m_taPhaseRot = hostToNumpy<std::complex<float>>((std::complex<float>*)taPhaseRot.addr(), CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps);

    return m_cfoEstVec;
}



} // namespace pycuphy
