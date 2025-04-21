/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <iostream>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "nvlog.h"
#include "cuphy.h"
#include "util.hpp"
#include "utils.cuh"
#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_ldpc.hpp"

namespace py = pybind11;

namespace pycuphy {


LdpcDerateMatch::LdpcDerateMatch(const bool scrambling, const cudaStream_t cuStream):
m_cuStream(cuStream) {

    // PUSCH rate match descriptors.
    // Descriptors hold kernel parameters in GPU.
    size_t dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxRateMatchGetDescrInfo(&m_dynDescrSizeBytes, &dynDescrAlignBytes);
    if(statusGetWorkspaceSize != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusGetWorkspaceSize, "cuphyPuschRxRateMatchGetDescrInfo");
    }
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);

    // Floating point config:
    // 0: FP32 in, FP32 out
    // 1: FP16 in, FP32 out
    // 2: FP32 in, FP16 out
    // 3: FP16 in, FP16 out
    // other values: invalid
    int fpConfig = 3;

    // Create the PUSCH rate match object.
    cuphyStatus_t statusCreate = cuphyCreatePuschRxRateMatch(&m_puschRmHndl, fpConfig, (int)scrambling);
    if(statusCreate != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusCreate, "cuphyCreatePuschRxRateMatch");
    }
}


LdpcDerateMatch::~LdpcDerateMatch() {
    destroy();
}


void LdpcDerateMatch::derateMatch(const std::vector<cuphy::tensor_ref>& llrs,
                                  void** deRmOutput,
                                  PuschParams& puschParams) {

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;

    std::vector<cuphyTensorPrm_t> inputLlrs(nUeGrps);
    for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ueGrpIdx++) {
        inputLlrs[ueGrpIdx].desc = llrs[ueGrpIdx].desc().handle();
        inputLlrs[ueGrpIdx].pAddr = (void*)llrs[ueGrpIdx].addr();
    }

    const PerTbParams* pTbPrmsCpu = puschParams.getPerTbPrmsCpuPtr();
    const PerTbParams* pTbPrmsGpu = puschParams.getPerTbPrmsGpuPtr();

    derateMatch(inputLlrs, deRmOutput, pTbPrmsCpu, pTbPrmsGpu, nUes);
}


void LdpcDerateMatch::derateMatch(const std::vector<cuphyTensorPrm_t>& inputLlrs,
                                  void** deRmOutput,
                                  const PerTbParams* pTbPrmsCpu,
                                  const PerTbParams* pTbPrmsGpu,
                                  int nUes) {
    int nUeGrps = inputLlrs.size();

    uint16_t nSchUes = 0;
    std::vector<uint16_t> schUserIdxsVec(nUes);
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        if(pTbPrmsCpu[ueIdx].isDataPresent) {
            schUserIdxsVec[nSchUes] = ueIdx;
            nSchUes++;
        }
    }

    // Launch config holds everything needed to launch kernel using CUDA driver API.
    cuphyPuschRxRateMatchLaunchCfg_t puschRmLaunchCfg;

    // Setup PUSCH rate match object.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t puschRmSetupStatus = cuphySetupPuschRxRateMatch(m_puschRmHndl,
                                                                  nSchUes,
                                                                  schUserIdxsVec.data(),
                                                                  pTbPrmsCpu,
                                                                  pTbPrmsGpu,
                                                                  (cuphyTensorPrm_t*)inputLlrs.data(),
                                                                  (cuphyTensorPrm_t*)inputLlrs.data(),
                                                                  deRmOutput,
                                                                  m_dynDescrBufCpu.addr(),
                                                                  m_dynDescrBufGpu.addr(),
                                                                  enableCpuToGpuDescrAsyncCpy,
                                                                  &puschRmLaunchCfg,
                                                                  m_cuStream);
    if(puschRmSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(puschRmSetupStatus, "cuphySetupPuschRxRateMatch");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(), m_dynDescrBufCpu.addr(), m_dynDescrSizeBytes, cudaMemcpyHostToDevice, m_cuStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Run PUSCH rate match.
    // Launch kernel using the CUDA driver API.
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = puschRmLaunchCfg.kernelNodeParamsDriver;
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


void LdpcDerateMatch::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxRateMatch(m_puschRmHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT,
                   "LdpcDerateMatch::destroy() failed to call cuphyDestroyPuschRxRateMatch()");
    }
}


PyLdpcDerateMatch::PyLdpcDerateMatch(const bool scrambling, const uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_linearAlloc(getBufferSize()),
m_derateMatch(scrambling, (cudaStream_t)cuStream)
{
    // Reserve pinned memory for the output addresses.
    CUDA_CHECK(cudaMallocHost(&m_deRmOutput, sizeof(void*) * MAX_N_TBS_SUPPORTED));
}


PyLdpcDerateMatch::~PyLdpcDerateMatch() {
    CUDA_CHECK_NO_THROW(cudaFreeHost(m_deRmOutput));
}


size_t PyLdpcDerateMatch::getBufferSize() const {
    const uint32_t EXTRA_PADDING = LINEAR_ALLOC_PAD_BYTES;
    uint32_t NUM_BYTES_PER_LLR = 2;
    size_t maxBytesRateMatch = NUM_BYTES_PER_LLR * MAX_N_TBS_SUPPORTED * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_RM_LLRS_PER_CB + EXTRA_PADDING;
    return maxBytesRateMatch;
}


const std::vector<py::array_t<float>>& PyLdpcDerateMatch::derateMatch(const std::vector<py::array_t<float>>& inputLlrs,
                                                                      const std::vector<uint32_t>& tbSizes,
                                                                      const std::vector<float>& codeRates,
                                                                      const std::vector<uint32_t>& rateMatchLengths,
                                                                      const std::vector<uint8_t>& qamMods,
                                                                      const std::vector<uint8_t>& numLayers,
                                                                      const std::vector<uint32_t>& rvs,
                                                                      const std::vector<uint32_t>& ndis,
                                                                      const std::vector<uint32_t>& cinits,
                                                                      const std::vector<uint32_t>& userGroupIdxs) {

    m_linearAlloc.reset();
    int nUeGrps = inputLlrs.size();
    int nUes = tbSizes.size();
    cuphy::buffer<PerTbParams, cuphy::pinned_alloc> tbPrmsCpu(nUes);
    cuphy::buffer<PerTbParams, cuphy::device_alloc> tbPrmsGpu(nUes);
    std::vector<cuphyTensorPrm_t> dInputLlrs(nUeGrps);
    m_pyDeRmOutput.clear();
    m_inputLlrTensors.resize(nUeGrps);

    for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ueGrpIdx++) {

        // Convert the input Numpy array to a device tensor.
        m_inputLlrTensors[ueGrpIdx] = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
            inputLlrs[ueGrpIdx],
            CUPHY_R_32F,
            CUPHY_R_16F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        dInputLlrs[ueGrpIdx].pAddr = m_inputLlrTensors[ueGrpIdx].addr();
        dInputLlrs[ueGrpIdx].desc = m_inputLlrTensors[ueGrpIdx].desc().handle();
    }

    cuphyLDPCParams ldpcParams;  // Dummy - values not actually used.

    // Count number of layers per UE group.
    std::vector<uint8_t> numUeGrpLayers(nUeGrps, 0);
    std::vector<std::vector<uint32_t>> layerMapArray(nUes);
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        layerMapArray[ueIdx].resize(numLayers[ueIdx]);
        for(int layerIdx = 0; layerIdx < numLayers[ueIdx]; layerIdx++) {
            layerMapArray[ueIdx][layerIdx] = numUeGrpLayers[userGroupIdxs[ueIdx]];
            numUeGrpLayers[userGroupIdxs[ueIdx]]++;
        }
    }

    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        setPerTbParams(tbPrmsCpu[ueIdx],
                       ldpcParams,
                       tbSizes[ueIdx],
                       codeRates[ueIdx],
                       qamMods[ueIdx],
                       ndis[ueIdx],
                       rvs[ueIdx],
                       rateMatchLengths[ueIdx],
                       cinits[ueIdx],
                       userGroupIdxs[ueIdx],
                       numLayers[ueIdx],
                       numUeGrpLayers[userGroupIdxs[ueIdx]],
                       layerMapArray[ueIdx]);
    }

    // Copy to GPU.
    CUDA_CHECK(cudaMemcpyAsync(tbPrmsGpu.addr(),
                               tbPrmsCpu.addr(),
                               sizeof(PerTbParams) * nUes,
                               cudaMemcpyHostToDevice,
                               m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Reserve output buffers.
    uint32_t NUM_BYTES_PER_LLR = 2;
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        size_t nBytesDeRm = NUM_BYTES_PER_LLR * tbPrmsCpu[ueIdx].Ncb_padded * tbPrmsCpu[ueIdx].num_CBs;
        m_deRmOutput[ueIdx] = m_linearAlloc.alloc(nBytesDeRm);
    }

    // Run the derate matching.
    const PerTbParams* pTbPrmsCpu = tbPrmsCpu.addr();
    const PerTbParams* pTbPrmsGpu = tbPrmsGpu.addr();
    m_derateMatch.derateMatch(dInputLlrs, m_deRmOutput, pTbPrmsCpu, pTbPrmsGpu, nUes);

    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        // Convert to host memory and return the Numpy array.
        uint32_t dim0 = tbPrmsCpu[ueIdx].Ncb_padded;
        uint32_t dim1 = tbPrmsCpu[ueIdx].num_CBs;

        cuphy::tensor_device dOutputTensor = cuphy::tensor_device(m_deRmOutput[ueIdx], CUPHY_R_16F, dim0, dim1, cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hOutputTensor = cuphy::tensor_pinned(CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
        hOutputTensor.convert(dOutputTensor, m_cuStream);
        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

        // Create the Numpy array for output.
        m_pyDeRmOutput.push_back(py::array_t<float>(
            {tbPrmsCpu[ueIdx].Ncb + 2 * tbPrmsCpu[ueIdx].Zc, dim1},  // Shape
            {sizeof(float), sizeof(float) * tbPrmsCpu[ueIdx].Ncb_padded},  // Strides (in bytes) for each index
            (float*)hOutputTensor.addr()
        ));
    }

    return m_pyDeRmOutput;
}

}  // namespace pycuphy
