/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#include "util.hpp"
#include "utils.cuh"
#include "cuphy.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_ldpc.hpp"

namespace pycuphy {

LdpcDecoder::LdpcDecoder(const cudaStream_t cuStream):
m_ctx(),
m_decoder(m_ctx),
m_cuStream(cuStream),
m_linearAlloc(getBufferSize()),
m_normalizationFactor(0.8125f) {}


size_t LdpcDecoder::getBufferSize() const {
    const int32_t EXTRA_PADDING = LINEAR_ALLOC_PAD_BYTES;

    const int32_t OUT_STRIDE_WORDS = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
    const int32_t BYTES_PER_WORD = 4;
    size_t ldpcOutputBufferSize = BYTES_PER_WORD * OUT_STRIDE_WORDS * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED + EXTRA_PADDING;

    size_t softOutputBufferSize = sizeof(__half) * MAX_DECODED_CODE_BLOCK_BIT_SIZE * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_TBS_SUPPORTED + EXTRA_PADDING;

    size_t nBytes = ldpcOutputBufferSize + softOutputBufferSize;
    return nBytes;
}


const std::vector<void*>& LdpcDecoder::getSoftOutputs() const {
    return m_ldpcSoftOutput;
}


void* LdpcDecoder::decode(void** deRmOutput, PuschParams& puschParams) {

    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;
    std::vector<void*> deRmLlr(deRmOutput, deRmOutput + nUes);

    const cuphyLDPCParams& ldpcParams = puschParams.getLdpcPrms();
    const PerTbParams* pTbPrmsCpu = puschParams.getPerTbPrmsCpuPtr();
    std::vector<PerTbParams> tbPrmsCpu(pTbPrmsCpu, pTbPrmsCpu + nUes);

    return decode(deRmLlr, tbPrmsCpu, ldpcParams);
}


void* LdpcDecoder::decode(const std::vector<void*>& deRmLlr,
                          const std::vector<PerTbParams>& tbPrmsCpu,
                          const cuphyLDPCParams& ldpcParams) {

    m_linearAlloc.reset();
    int nUes = tbPrmsCpu.size();
    m_ldpcSoftOutput.resize(nUes);

    int dims[2];
    int strides[2];
    const int32_t OUT_STRIDE_WORDS = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
    const int32_t BYTES_PER_WORD = 4;

    // Allocate output memory.
    size_t lpdcDecodeOutSize = 0;
    for(uint32_t tbIdx = 0; tbIdx < nUes; tbIdx++) {
        lpdcDecodeOutSize += BYTES_PER_WORD * OUT_STRIDE_WORDS * tbPrmsCpu[tbIdx].num_CBs;
    }
    m_ldpcOutput = m_linearAlloc.alloc(lpdcDecodeOutSize);

    size_t outputBytes = 0;
    for(uint32_t tbIdx = 0; tbIdx < nUes; tbIdx++) {
        cuphy::LDPC_decode_config decoderConfig(ldpcParams.useHalf ? CUPHY_R_16F : CUPHY_R_32F,
                                                ldpcParams.parityNodesArray[tbIdx],
                                                tbPrmsCpu[tbIdx].Zc,
                                                ldpcParams.nIterations,
                                                ldpcParams.KbArray[tbIdx],
                                                m_normalizationFactor,
                                                ldpcParams.flags,
                                                tbPrmsCpu[tbIdx].bg,
                                                ldpcParams.algoIndex,
                                                nullptr);
        if(m_normalizationFactor <= 0.0f) {
            m_decoder.set_normalization(decoderConfig);
        }

        size_t numElems = round_up_to_next(ldpcParams.KbArray[tbIdx] * tbPrmsCpu[tbIdx].Zc, (unsigned int)2);
        size_t ldpcSoftOutputSize = sizeof(__half) * numElems * tbPrmsCpu[tbIdx].num_CBs;
        m_ldpcSoftOutput[tbIdx] = m_linearAlloc.alloc(ldpcSoftOutputSize);

        dims[0]    = tbPrmsCpu[tbIdx].Ncb_padded;
        dims[1]    = tbPrmsCpu[tbIdx].num_CBs;
        strides[0] = 1;
        strides[1] = tbPrmsCpu[tbIdx].Ncb_padded;

        cuphy::tensor_layout tlInput(2, dims, strides);
        cuphy::tensor_info   tiInput(ldpcParams.useHalf ? CUPHY_R_16F : CUPHY_R_32F, tlInput);
        cuphy::tensor_desc   tdInput(tiInput, cuphy::tensor_flags::align_tight);

        dims[0]    = numElems;
        dims[1]    = tbPrmsCpu[tbIdx].num_CBs;
        strides[0] = 1;
        strides[1] = numElems;

        cuphy::tensor_layout tlSoftOutput(2, dims, strides);
        cuphy::tensor_info   tiSoftOutput(CUPHY_R_16F, tlSoftOutput);
        cuphy::tensor_desc   tdSoftOutput(tiSoftOutput, cuphy::tensor_flags::align_tight);

        dims[0]    = MAX_DECODED_CODE_BLOCK_BIT_SIZE;
        dims[1]    = tbPrmsCpu[tbIdx].num_CBs;
        strides[0] = 1;
        strides[1] = MAX_DECODED_CODE_BLOCK_BIT_SIZE;

        cuphy::tensor_layout tlOutput(2, dims, strides);
        cuphy::tensor_info   tiOutput(CUPHY_BIT, tlOutput);
        cuphy::tensor_desc   tdOutput(tiOutput, cuphy::tensor_flags::align_tight);

        cuphy::LDPC_decode_tensor_params decoderTensor(
            decoderConfig,                                          // LDPC configuration
            tdOutput.handle(),                                      // output descriptor
            static_cast<uint8_t*>(m_ldpcOutput) + outputBytes,      // output address
            tdInput.handle(),                                       // LLR descriptor
            deRmLlr[tbIdx],                                         // LLR address
            tdSoftOutput.handle(),                                  // Soft output descriptor
            m_ldpcSoftOutput[tbIdx]);                               // Soft output address

        // Run the decoder for this TB.
        m_decoder.decode(decoderTensor, m_cuStream);

        outputBytes += tdOutput.get_size_in_bytes();
    }

    return m_ldpcOutput;
}


PyLdpcDecoder::PyLdpcDecoder(const uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_decoder((cudaStream_t)cuStream) {
    m_ldpcParams.nIterations = 10;
    m_ldpcParams.earlyTermination = false;
    m_ldpcParams.algoIndex = 0;
    m_ldpcParams.flags = CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT | CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    m_ldpcParams.useHalf = true;
}


void PyLdpcDecoder::setNumIterations(const uint32_t numIterations) {
    m_ldpcParams.nIterations = numIterations;
}


void PyLdpcDecoder::setThroughputMode(const uint8_t throughputMode) {
    uint8_t mode = throughputMode ? CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT : 0;
    m_ldpcParams.flags = mode | CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;

}


const std::vector<py::array_t<float>>& PyLdpcDecoder::decode(const std::vector<py::array_t<float>>& inputLlrs,
                                                             const std::vector<uint32_t>& tbSizes,
                                                             const std::vector<float>& codeRates,
                                                             const std::vector<uint32_t>& rvs,
                                                             const std::vector<uint32_t>& rateMatchLengths) {
    int nUes = inputLlrs.size();
    m_tbParams.resize(nUes);
    std::vector<void*> dInputLlrs(nUes);
    m_ldpcOutput.clear();
    m_softOutput.clear();

    m_inputLlrTensors.resize(nUes);
    m_ldpcParams.parityNodesArray.clear();
    m_ldpcParams.KbArray.clear();

    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        // Convert input numpy array to tensor.
        m_inputLlrTensors[ueIdx] = deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
            inputLlrs[ueIdx],
            CUPHY_R_32F,
            CUPHY_R_16F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        dInputLlrs[ueIdx] = m_inputLlrTensors[ueIdx].addr();

        setPerTbParams(m_tbParams[ueIdx],
                       m_ldpcParams,
                       tbSizes[ueIdx],
                       codeRates[ueIdx],
                       2,              // qamModOrder not used here.
                       1,              // ndi not used here.
                       rvs[ueIdx],
                       rateMatchLengths[ueIdx],
                       0,  // cinit not used.
                       0,  // userGroupIndex not used.
                       1,  // Number of layers not used
                       1,
                       {0} // Layer mapping not used
                       );
    }

    void* ldpcOutput = m_decoder.decode(dInputLlrs, m_tbParams, m_ldpcParams);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    size_t dims[2];
    size_t strides[2];
    size_t outputBytes = 0;
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        // LDPC output tensor layout.
        dims[0]    = MAX_DECODED_CODE_BLOCK_BIT_SIZE;
        dims[1]    = m_tbParams[ueIdx].num_CBs;
        strides[0] = 1;
        strides[1] = MAX_DECODED_CODE_BLOCK_BIT_SIZE;

        cuphy::tensor_device dOutputTensor = cuphy::tensor_device(static_cast<uint8_t*>(ldpcOutput) + outputBytes,
                                                                  CUPHY_BIT,
                                                                  dims[0],
                                                                  dims[1],
                                                                  cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hOutputTensor = cuphy::tensor_pinned(CUPHY_R_32F, dims[0], dims[1], cuphy::tensor_flags::align_tight);
        hOutputTensor.convert(dOutputTensor, m_cuStream);

        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

        // Create the Numpy array for output.
        m_ldpcOutput.push_back(py::array_t<float>(
            dims,  // Shape
            {sizeof(float), sizeof(float) * MAX_DECODED_CODE_BLOCK_BIT_SIZE},  // Strides (in bytes) for each index
            (float*)hOutputTensor.addr()
        ));

        outputBytes += dOutputTensor.desc().get_size_in_bytes();
    }

    return m_ldpcOutput;
}


const std::vector<py::array_t<float>>& PyLdpcDecoder::getSoftOutputs() {

    m_softOutput.clear();
    const std::vector<void*> ldpcSoftOutput = m_decoder.getSoftOutputs();

    int nUes = ldpcSoftOutput.size();

    size_t dims[2];
    size_t strides[2];
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {

        // LDPC soft output tensor layout.
        size_t numElems = round_up_to_next(m_ldpcParams.KbArray[ueIdx] * m_tbParams[ueIdx].Zc, (unsigned int)2);
        dims[0]    = numElems;
        dims[1]    = m_tbParams[ueIdx].num_CBs;
        strides[0] = 1;
        strides[1] = numElems;

        cuphy::tensor_device dOutputTensor = cuphy::tensor_device(ldpcSoftOutput[ueIdx], m_ldpcParams.useHalf ? CUPHY_R_16F : CUPHY_R_32F, dims[0], dims[1], cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hOutputTensor = cuphy::tensor_pinned(CUPHY_R_32F, dims[0], dims[1], cuphy::tensor_flags::align_tight);
        hOutputTensor.convert(dOutputTensor, m_cuStream);

        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

        // Create the Numpy array for output.
        m_softOutput.push_back(py::array_t<float>(
            {dims[0], dims[1]},  // Shape
            {sizeof(float), sizeof(float) * dims[0]},  // Strides (in bytes) for each index
            (float*)hOutputTensor.addr()
        ));
    }

    return m_softOutput;
}


} // namespace pycuphy
