/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#include "pycuphy_debug.hpp"
#include "pycuphy_rsrp.hpp"


namespace py = pybind11;


namespace pycuphy {

RsrpEstimator::~RsrpEstimator() {
    try {
        destroy();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception from RsrpEstimator destructor: " << e.what() << "\n";
    }
}


RsrpEstimator::RsrpEstimator(cudaStream_t cuStream):
m_linearAlloc(getBufferSize()),
m_cuStream(cuStream) {

    // Allocate descriptors.
    allocateDescr();

    cuphyStatus_t status = cuphyCreatePuschRxRssi(&m_puschRxRssiHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxRssi()");
    }
}


size_t RsrpEstimator::getBufferSize() const {

    static constexpr uint32_t N_BYTES_R32 = sizeof(cuphy::type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t MAX_N_UE = MAX_N_TBS_SUPPORTED;
    static constexpr uint32_t N_BYTES_PER_UINT32 = 4;

    uint32_t maxBytesRsrp                = N_BYTES_R32 * MAX_N_UE + LINEAR_ALLOC_PAD_BYTES;
    uint32_t maxBytesNoiseVarPostEq      = N_BYTES_R32 * MAX_N_UE + LINEAR_ALLOC_PAD_BYTES;
    // Post-eq SINR and pre-eq SINR
    uint32_t maxBytesSinr                = (2 * N_BYTES_R32 * MAX_N_UE) + LINEAR_ALLOC_PAD_BYTES;
    uint32_t maxBytesRsrpInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;

    size_t nBytesBuffer = maxBytesRsrp + maxBytesNoiseVarPostEq + maxBytesSinr + maxBytesRsrpInterCtaSyncCnt;
    return nBytesBuffer;
}


void RsrpEstimator::debugDump(H5DebugDump& debugDump, cudaStream_t cuStream) {
    debugDump.dump("SinrPreEq", m_tSinrPreEq, cuStream);
    debugDump.dump("SinrPostEq", m_tSinrPostEq, cuStream);
    debugDump.dump("Rsrp", m_tRsrp, cuStream);
}


void RsrpEstimator::allocateDescr() {

    size_t dynDescSizeBytesRssi, dynDescrAlignBytesRssi;
    size_t dynDescrAlignBytesRsrp;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxRssiGetDescrInfo(&dynDescSizeBytesRssi,
                                                                        &dynDescrAlignBytesRssi,
                                                                        &m_dynDescrSizeBytes,
                                                                        &dynDescrAlignBytesRsrp);
    if(statusGetWorkspaceSize != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusGetWorkspaceSize, "cuphyPuschRxRssiGetDescrInfo()");
    }

    m_dynDescrSizeBytes = ((m_dynDescrSizeBytes + (dynDescrAlignBytesRsrp - 1)) / dynDescrAlignBytesRsrp) * dynDescrAlignBytesRsrp;
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);
}


void RsrpEstimator::estimate(PuschParams& puschParams) {

    m_linearAlloc.reset();

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;

    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    std::vector<cuphy::tensor_device>& tDataRx = puschParams.getDataTensor();

    // Allocate output tensor arrays in device memory.
    m_tNoiseIntfVarPostEq.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tNoiseIntfVarPostEq);

    m_tRsrp.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tRsrp);

    m_tSinrPreEq.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tSinrPreEq);

    m_tSinrPostEq.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tSinrPostEq);

    m_tInterCtaSyncCnt.desc().set(CUPHY_R_32U, nUeGrps, cuphy::tensor_flags::align_tight);
    m_linearAlloc.alloc(m_tInterCtaSyncCnt);

    m_linearAlloc.memset(0, m_cuStream);

    for(int i = 0; i < nUeGrps; ++i) {
        uint16_t cellPrmDynIdx = puschParams.m_puschDynPrms.pCellGrpDynPrm->pUeGrpPrms[i].pCellPrm->cellPrmDynIdx;
        copyTensorData(tDataRx[cellPrmDynIdx], pPuschRxUeGrpPrmsCpu[i].tInfoDataRx);

        copyTensorData(m_tNoiseIntfVarPostEq, pPuschRxUeGrpPrmsCpu[i].tInfoNoiseVarPostEq);
        copyTensorData(m_tRsrp, pPuschRxUeGrpPrmsCpu[i].tInfoRsrp);
        copyTensorData(m_tSinrPreEq, pPuschRxUeGrpPrmsCpu[i].tInfoSinrPreEq);
        copyTensorData(m_tSinrPostEq, pPuschRxUeGrpPrmsCpu[i].tInfoSinrPostEq);
        copyTensorData(m_tInterCtaSyncCnt, pPuschRxUeGrpPrmsCpu[i].tInfoRsrpInterCtaSyncCnt);
    }

    // Run setup.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyPuschRxRsrpLaunchCfgs_t rsrpLaunchCfgs;
    rsrpLaunchCfgs.nCfgs = CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS;
    cuphyStatus_t setupRsrpStatus = cuphySetupPuschRxRsrp(m_puschRxRssiHndl,
                                                          pPuschRxUeGrpPrmsCpu,
                                                          pPuschRxUeGrpPrmsGpu,
                                                          nUeGrps,
                                                          puschParams.getMaxNumPrb(),
                                                          CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS,
                                                          enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                          static_cast<void*>(m_dynDescrBufCpu.addr()),
                                                          static_cast<void*>(m_dynDescrBufGpu.addr()),
                                                          &rsrpLaunchCfgs,
                                                          m_cuStream);
    if(setupRsrpStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(setupRsrpStatus, "cuphySetupPuschRxRsrp()");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(),
                                   m_dynDescrBufCpu.addr(),
                                   m_dynDescrSizeBytes,
                                   cudaMemcpyHostToDevice,
                                   m_cuStream));
        puschParams.copyPuschRxUeGrpPrms();
    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < rsrpLaunchCfgs.nCfgs; ++hetCfgIdx) {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = rsrpLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
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


void RsrpEstimator::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxRssi(m_puschRxRssiHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxRssi()");
    }
}


PyRsrpEstimator::PyRsrpEstimator(uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_rsrpEstimator((cudaStream_t)cuStream) {
    m_tChannelEst.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tReeDiagInv.resize(MAX_N_USER_GROUPS_SUPPORTED);
}


const py::array_t<float>& PyRsrpEstimator::estimate(const std::vector<py::array_t<std::complex<float>>>& chEst,
                                                    const std::vector<py::array_t<float>>& reeDiagInv,
                                                    const py::array& infoNoiseVarPreEq,
                                                    PuschParams& puschParams) {

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsCpu = puschParams.getPuschRxUeGrpPrmsCpuPtr();
    cuphyPuschRxUeGrpPrms_t* pPuschRxUeGrpPrmsGpu = puschParams.getPuschRxUeGrpPrmsGpuPtr();

    // Read inputs.
    m_tInfoNoiseVarPreEq = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
        infoNoiseVarPreEq,
        CUPHY_R_32F,
        CUPHY_R_32F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);


    for(int i = 0; i < nUeGrps; ++i) {
        m_tChannelEst[i] = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
            chEst[i],
            CUPHY_C_32F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        copyTensorData(m_tChannelEst[i], pPuschRxUeGrpPrmsCpu[i].tInfoHEst);

        m_tReeDiagInv[i] = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
            reeDiagInv[i],
            CUPHY_R_32F,
            CUPHY_R_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        copyTensorData(m_tReeDiagInv[i], pPuschRxUeGrpPrmsCpu[i].tInfoReeDiagInv);

        copyTensorData(m_tInfoNoiseVarPreEq, pPuschRxUeGrpPrmsCpu[i].tInfoNoiseVarPreEq);
    }

    // Run the estimator.
    m_rsrpEstimator.estimate(puschParams);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Get the return values.
    const cuphy::tensor_ref& dRsrp = m_rsrpEstimator.getRsrp();
    const cuphy::tensor_ref& dNoiseIntVarPostEq = m_rsrpEstimator.getNoiseIntVarPostEq();
    const cuphy::tensor_ref& dSinrPreEq = m_rsrpEstimator.getSinrPreEq();
    const cuphy::tensor_ref& dSinrPostEq = m_rsrpEstimator.getSinrPostEq();

    // Move the return values to host and Python.
    cuphy::tensor_pinned rsrp = cuphy::tensor_pinned(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    rsrp.convert(dRsrp, m_cuStream);

    cuphy::tensor_pinned infoNoiseVarPostEq = cuphy::tensor_pinned(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    infoNoiseVarPostEq.convert(dNoiseIntVarPostEq, m_cuStream);

    cuphy::tensor_pinned sinrPreEq = cuphy::tensor_pinned(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    sinrPreEq.convert(dSinrPreEq, m_cuStream);

    cuphy::tensor_pinned sinrPostEq = cuphy::tensor_pinned(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
    sinrPostEq.convert(dSinrPostEq, m_cuStream);

    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    m_rsrp = hostToNumpy<float>((float*)rsrp.addr(), nUes);
    m_infoNoiseVarPostEq = hostToNumpy<float>((float*)infoNoiseVarPostEq.addr(), nUes);
    m_sinrPreEq = hostToNumpy<float>((float*)sinrPreEq.addr(), nUes);
    m_sinrPostEq = hostToNumpy<float>((float*)sinrPostEq.addr(), nUes);

    return m_rsrp;
}



} // namespace pycuphy
