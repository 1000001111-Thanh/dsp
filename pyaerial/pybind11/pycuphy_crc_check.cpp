/* Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "pusch_utils.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_crc_check.hpp"

namespace pycuphy {


CrcChecker::CrcChecker(const cudaStream_t cuStream):
m_cuStream(cuStream),
m_linearAlloc(getBufferSize()) {

    allocateDescr();

    cuphyStatus_t status = cuphyCreatePuschRxCrcDecode(&m_crcDecodeHndl, 1);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxCrcDecode()");
    }
}

CrcChecker::~CrcChecker() {
    destroy();
}

size_t CrcChecker::getBufferSize() const {

    const uint32_t N_BYTES_PER_UINT32 = 4;
    const uint32_t BYTES_PER_WORD = 4;
    const uint32_t OUT_STRIDE_WORDS = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
    const uint32_t EXTRA_PADDING = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES;


    size_t nBytesBuffer = 0;

    uint32_t maxBytesCbCrc = N_BYTES_PER_UINT32 * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED;
    nBytesBuffer += maxBytesCbCrc + EXTRA_PADDING;

    uint32_t maxBytesTbCrc = N_BYTES_PER_UINT32 * MAX_N_TBS_SUPPORTED;
    nBytesBuffer += maxBytesTbCrc + EXTRA_PADDING;

    uint32_t maxBytesOut = BYTES_PER_WORD * OUT_STRIDE_WORDS * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED;
    nBytesBuffer += maxBytesOut  + EXTRA_PADDING;

    return nBytesBuffer;
}


void CrcChecker::allocateDescr() {

    size_t dynDescrAlignBytes;
    cuphyStatus_t status = cuphyPuschRxCrcDecodeGetDescrInfo(&m_dynDescrSizeBytes, &dynDescrAlignBytes);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxCrcDecodeGetDescrInfo()");
    }
    m_dynDescrSizeBytes = ((m_dynDescrSizeBytes + (dynDescrAlignBytes - 1)) / dynDescrAlignBytes) * dynDescrAlignBytes;
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);
}


void CrcChecker::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxCrcDecode(m_crcDecodeHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxCrcDecode()");
    }
}


void CrcChecker::checkCrc(void* ldpcOutput,
                          PuschParams& puschParams) {
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;
    const PerTbParams* pTbPrmsCpu = puschParams.getPerTbPrmsCpuPtr();
    const PerTbParams* pTbPrmsGpu = puschParams.getPerTbPrmsGpuPtr();
    checkCrc(ldpcOutput, pTbPrmsCpu, pTbPrmsGpu, nUes);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));
    uint32_t*   pTbCrcsHost     = puschParams.m_puschDynPrms.pDataOut->pTbCrcs;
    uint32_t    totNumTbs       = puschParams.m_puschDynPrms.pDataOut->totNumTbs;
    CUDA_CHECK(cudaMemcpyAsync(pTbCrcsHost, m_outputTbCrcs, sizeof(uint32_t) * totNumTbs, cudaMemcpyDeviceToHost, m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));
}


void CrcChecker::checkCrc(void* ldpcOutput,
                          const PerTbParams* tbPrmsCpu,
                          const PerTbParams* tbPrmsGpu,
                          const int nUes)
{
    m_linearAlloc.reset();

    m_totNumTbs = 0;
    m_totNumCbs = 0;
    m_totNumPayloadBytes = 0;
    uint16_t nSchUes = 0;
    std::vector<uint16_t> schUserIdxsVec(nUes);
    m_tbPayloadStartOffsets.resize(nUes);
    m_tbCrcStartOffsets.resize(nUes);
    m_cbCrcStartOffsets.resize(nUes);
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        if(tbPrmsCpu[ueIdx].isDataPresent) {
            schUserIdxsVec[nSchUes] = ueIdx;
            nSchUes += 1;

            m_tbPayloadStartOffsets[ueIdx] = m_totNumPayloadBytes;
            m_tbCrcStartOffsets[ueIdx] = m_totNumTbs;
            m_cbCrcStartOffsets[ueIdx] = m_totNumCbs;

            m_totNumTbs += 1;
            m_totNumCbs += tbPrmsCpu[ueIdx].num_CBs;

            uint8_t crcSizeBytes = tbPrmsCpu[ueIdx].tbSize > 3824 ? 3 : 2;     // 38.212, section 7.2.1
            uint32_t tbSizeBytes = tbPrmsCpu[ueIdx].tbSize / 8 + crcSizeBytes; // in cuPHY each TB includes TB payload + TB CRC
            m_totNumPayloadBytes += tbSizeBytes;

            uint32_t tbWordAlignPaddingBytes = (sizeof(uint32_t) - (tbSizeBytes % sizeof(uint32_t))) % sizeof(uint32_t);
            m_totNumPayloadBytes += tbWordAlignPaddingBytes;
        }
    }

    // Allocate memory for outputs.
    m_outputTbCrcs = static_cast<uint32_t*>(m_linearAlloc.alloc(m_totNumTbs * sizeof(uint32_t)));
    m_outputCbCrcs = static_cast<uint32_t*>(m_linearAlloc.alloc(m_totNumCbs * sizeof(uint32_t)));
    m_outputTbs = static_cast<uint8_t*>(m_linearAlloc.alloc(m_totNumPayloadBytes * sizeof(uint8_t)));

    cuphyPuschRxCrcDecodeLaunchCfg_t crcLaunchCfgs[2];
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t setupCrcDecodeStatus = cuphySetupPuschRxCrcDecode(m_crcDecodeHndl,
                                                                    nSchUes,
                                                                    schUserIdxsVec.data(),
                                                                    m_outputCbCrcs,
                                                                    m_outputTbs,
                                                                    static_cast<uint32_t*>(ldpcOutput),
                                                                    m_outputTbCrcs,
                                                                    tbPrmsCpu,
                                                                    tbPrmsGpu,
                                                                    m_dynDescrBufCpu.addr(),
                                                                    m_dynDescrBufGpu.addr(),
                                                                    enableCpuToGpuDescrAsyncCpy,
                                                                    &crcLaunchCfgs[0],
                                                                    &crcLaunchCfgs[1],
                                                                    m_cuStream);
    if(setupCrcDecodeStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(setupCrcDecodeStatus, "cuphySetupPuschRxCrcDecode");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(), m_dynDescrBufCpu.addr(), m_dynDescrSizeBytes, cudaMemcpyHostToDevice, m_cuStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Run code block CRC decoding.
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver1 = crcLaunchCfgs[0].kernelNodeParamsDriver;
    CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(kernelNodeParamsDriver1.func,
                                                     kernelNodeParamsDriver1.gridDimX,
                                                     kernelNodeParamsDriver1.gridDimY,
                                                     kernelNodeParamsDriver1.gridDimZ,
                                                     kernelNodeParamsDriver1.blockDimX,
                                                     kernelNodeParamsDriver1.blockDimY,
                                                     kernelNodeParamsDriver1.blockDimZ,
                                                     kernelNodeParamsDriver1.sharedMemBytes,
                                                     static_cast<CUstream>(m_cuStream),
                                                     kernelNodeParamsDriver1.kernelParams,
                                                     kernelNodeParamsDriver1.extra));


    // Run transport block CRC decoding.
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver2 = crcLaunchCfgs[1].kernelNodeParamsDriver;
    CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(kernelNodeParamsDriver2.func,
                                                     kernelNodeParamsDriver2.gridDimX,
                                                     kernelNodeParamsDriver2.gridDimY,
                                                     kernelNodeParamsDriver2.gridDimZ,
                                                     kernelNodeParamsDriver2.blockDimX,
                                                     kernelNodeParamsDriver2.blockDimY,
                                                     kernelNodeParamsDriver2.blockDimZ,
                                                     kernelNodeParamsDriver2.sharedMemBytes,
                                                     static_cast<CUstream>(m_cuStream),
                                                     kernelNodeParamsDriver2.kernelParams,
                                                     kernelNodeParamsDriver2.extra));

}


PyCrcChecker::PyCrcChecker(const uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_crcChecker((cudaStream_t)cuStream)
{}

const std::vector<py::array_t<uint8_t>>& PyCrcChecker::checkCrc(const py::array_t<float>& ldpcOutput,
                                                                const std::vector<uint32_t>& tbSizes,
                                                                const std::vector<float>& codeRates) {
    int nUes = tbSizes.size();
    cuphy::buffer<PerTbParams, cuphy::pinned_alloc> tbPrmsCpu(nUes);
    cuphy::buffer<PerTbParams, cuphy::device_alloc> tbPrmsGpu(nUes);

    // Move input bits to device.
    cuphy::tensor_device tCrcInput = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
        ldpcOutput,
        CUPHY_R_32F,
        CUPHY_BIT,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    void* pCrcInput = tCrcInput.addr();

    // Set TB params.
    cuphyLDPCParams ldpcParams;  // Dummy - values not actually used.
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        setPerTbParams(tbPrmsCpu[ueIdx],
                       ldpcParams,
                       tbSizes[ueIdx],
                       codeRates[ueIdx],
                       2,              // qamModOrder not used here.
                       1,              // ndi not used here.
                       0,              // rv not used here.
                       0,              // rate matching length not used.
                       0,              // cinit not used.
                       0,              // userGroupIdx not used here.
                       1,              // numLayers not used here.
                       1,
                       {0});
    }

    // Copy to GPU.
    CUDA_CHECK(cudaMemcpyAsync(tbPrmsGpu.addr(),
                               tbPrmsCpu.addr(),
                               sizeof(PerTbParams) * nUes,
                               cudaMemcpyHostToDevice,
                               m_cuStream));

    // Run the CRC checking kernels.
    const PerTbParams* pTbPrmsCpu = tbPrmsCpu.addr();
    const PerTbParams* pTbPrmsGpu = tbPrmsGpu.addr();
    m_crcChecker.checkCrc(pCrcInput, pTbPrmsCpu, pTbPrmsGpu, nUes);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Move outputs to host.
    const uint8_t* pTbPayloads = m_crcChecker.getOutputTbs();
    const uint32_t* pCbCrcs = m_crcChecker.getCbCrcs();
    const uint32_t* pTbCrcs = m_crcChecker.getTbCrcs();

    uint32_t totNumTbs = m_crcChecker.getTotNumTbs();
    uint32_t totNumCbs = m_crcChecker.getTotNumCbs();
    uint32_t totNumPayloadBytes = m_crcChecker.getTotNumPayloadBytes();

    cuphy::tensor_device dTbPayloads = cuphy::tensor_device((void*)pTbPayloads, CUPHY_R_8U, totNumPayloadBytes, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned hTbPayloads = cuphy::tensor_pinned(CUPHY_R_8U, totNumPayloadBytes, cuphy::tensor_flags::align_tight);
    hTbPayloads.convert(dTbPayloads, m_cuStream);

    cuphy::tensor_device dTbCrcs = cuphy::tensor_device((void*)pTbCrcs, CUPHY_R_32U, totNumTbs, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned hTbCrcs = cuphy::tensor_pinned(CUPHY_R_32U, totNumTbs, cuphy::tensor_flags::align_tight);
    hTbCrcs.convert(dTbCrcs, m_cuStream);

    cuphy::tensor_device dCbCrcs = cuphy::tensor_device((void*)pCbCrcs, CUPHY_R_32U, totNumCbs, cuphy::tensor_flags::align_tight);
    cuphy::tensor_pinned hCbCrcs = cuphy::tensor_pinned(CUPHY_R_32U, totNumCbs, cuphy::tensor_flags::align_tight);
    hCbCrcs.convert(dCbCrcs, m_cuStream);

    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // And then move to Numpy.
    const std::vector<uint32_t>& tbPayloadStartOffsets = m_crcChecker.getTbPayloadStartOffsets();
    const std::vector<uint32_t>& tbCrcStartOffsets = m_crcChecker.getTbCrcStartOffsets();
    const std::vector<uint32_t>& cbCrcStartOffsets = m_crcChecker.getCbCrcStartOffsets();

    m_tbPayloads.resize(nUes);
    m_tbCrcs.resize(nUes);
    m_cbCrcs.resize(nUes);
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        m_tbPayloads[ueIdx] = py::array_t<uint8_t>({tbPrmsCpu[ueIdx].tbSize / 8}, {sizeof(uint8_t)}, (uint8_t*)hTbPayloads.addr() + tbPayloadStartOffsets[ueIdx]);
        m_tbCrcs[ueIdx] = py::array_t<uint32_t>({1}, {sizeof(uint32_t)}, (uint32_t*)hTbCrcs.addr() + tbCrcStartOffsets[ueIdx]);
        m_cbCrcs[ueIdx] = py::array_t<uint32_t>({tbPrmsCpu[ueIdx].num_CBs}, {sizeof(uint32_t)}, (uint32_t*)hCbCrcs.addr() + cbCrcStartOffsets[ueIdx]);

    }

    return m_tbPayloads;
}

} // namespace pycuphy
