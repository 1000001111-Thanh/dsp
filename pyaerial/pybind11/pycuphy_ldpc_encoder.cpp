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
#include "util.hpp"
#include "utils.cuh"
#include "cuphy_internal.h"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_util.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std::complex_literals;
namespace py = pybind11;


namespace pycuphy {


LdpcEncoder::LdpcEncoder(const uint64_t inputDevicePtr,
                         const uint64_t tempInputHostPtr,
                         const uint64_t outputDevicePtr,
                         const uint64_t outputHostPtr,
                         const uint64_t cuStream):
m_inputDevicePtr((void *)inputDevicePtr),
m_tempInputHostPtr((void *)tempInputHostPtr),
m_outputDevicePtr((void *)outputDevicePtr),
m_outputHostPtr((void *)outputHostPtr),
m_numIterations(0),
m_puncture(1),
m_cuStream((cudaStream_t)cuStream)
{}


void LdpcEncoder::setProfilingIterations(const uint16_t numIterations) {
    m_numIterations = numIterations;
}

void LdpcEncoder::setPuncturing(const uint8_t puncture) {
    m_puncture = puncture;
}


py::array_t<float> LdpcEncoder::encode(const py::array& inputData,
                                       const uint32_t tbSize,
                                       const float codeRate,
                                       const int rv) {
    m_tbParams.resize(1);  // TODO: Support more.

    const uint32_t ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits

    if (!m_tempInputHostPtr || !m_inputDevicePtr || !m_outputDevicePtr || !m_outputHostPtr) {
        throw std::runtime_error("LdpcEncoder::encode: Memory not allocated!");
    }

    // Access the input data address.
    py::array_t<float, py::array::c_style | py::array::forcecast> inputArray = inputData;
    py::buffer_info buf = inputArray.request();
    uint32_t C = buf.shape[1];
    uint32_t K = buf.shape[0];

    // Set the encoder parameters into the struct.
    setPdschPerTbParams(m_tbParams[0],
                        tbSize,
                        codeRate,
                        0,        // rateMatchLen not used
                        2,        // qamMod not used
                        C,        // numCodeBlocks
                        0,        // numCodedBits computed
                        rv,
                        1,        // numLayers not used
                        0);       // cinit not used

    // Parameter selection based on base graph.
    uint16_t maxParityNodes;
    int nCwNodes;
    if(m_tbParams[0].bg == 1) {
        maxParityNodes = CUPHY_LDPC_MAX_BG1_PARITY_NODES;
        nCwNodes = CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES;
    }
    else {
        maxParityNodes = CUPHY_LDPC_MAX_BG2_PARITY_NODES;
        nCwNodes = CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES;
    }
    uint32_t effN = m_tbParams[0].Zc * (nCwNodes + (m_puncture ? 0 : 2));

    // Convert the float array data to 32 bit array data. Store under m_tempInputHostPtr.
    fromNumpyBitArray<uint32_t>((float*)buf.ptr, (uint32_t *)m_tempInputHostPtr, K, C);

    // Tensor to hold input uncoded data in device memory.
    uint32_t roundedK = round_up_to_next(K, ELEMENT_SIZE);
    cuphy::tensor_device dInputTensor(m_inputDevicePtr, CUPHY_BIT, roundedK, C);
    CUDA_CHECK(cudaMemcpyAsync(dInputTensor.addr(), m_tempInputHostPtr, dInputTensor.desc().get_size_in_bytes(), cudaMemcpyHostToDevice, m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Output tensor in device memory.
    uint32_t roundedN = round_up_to_next(effN, ELEMENT_SIZE);
    cuphy::tensor_device dOutputTensor(m_outputDevicePtr, CUPHY_BIT, roundedN, C);

    // Allocate launch config struct.
    std::unique_ptr<cuphyLDPCEncodeLaunchConfig> ldpcHandle = std::make_unique<cuphyLDPCEncodeLaunchConfig>();

    // Allocate descriptors and setup LDPC encoder.
    uint8_t descAsyncCopy = 1; // Copy descriptor to the GPU during setup
    size_t  workspaceSize = 0;
    int maxUes = PDSCH_MAX_UES_PER_CELL_GROUP;

    size_t descSize = 0, allocSize = 0;
    cuphyStatus_t status = cuphyLDPCEncodeGetDescrInfo(&descSize, &allocSize, maxUes, &workspaceSize);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LdpcEncoder::encode: cuphyLDPCEncodeGetDescrInfo error!");
    }

    cuphy::unique_device_ptr<uint8_t> dLdpcDesc = cuphy::make_unique_device<uint8_t>(descSize);
    cuphy::unique_pinned_ptr<uint8_t> hLdpcDesc = cuphy::make_unique_pinned<uint8_t>(descSize);

    cuphy::unique_device_ptr<uint8_t> dWorkspace = cuphy::make_unique_device<uint8_t>(workspaceSize);
    cuphy::unique_pinned_ptr<uint8_t> hWorkspace = cuphy::make_unique_pinned<uint8_t>(workspaceSize);

    // Setup the LDPC Encoder
    status = cuphySetupLDPCEncode(ldpcHandle.get(),
                                  dInputTensor.desc().handle(),
                                  dInputTensor.addr(),
                                  dOutputTensor.desc().handle(),
                                  dOutputTensor.addr(),
                                  m_tbParams[0].bg,
                                  m_tbParams[0].Zc,
                                  m_puncture,
                                  maxParityNodes,
                                  m_tbParams[0].rv,
                                  0,
                                  1,
                                  nullptr,
                                  nullptr,
                                  hWorkspace.get(),
                                  dWorkspace.get(),
                                  hLdpcDesc.get(),
                                  dLdpcDesc.get(),
                                  descAsyncCopy,
                                  m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("LdpcEncoder::encode: Invalid argument(s) for cuphySetupLDPCEncode!");
    }

    if(!m_numIterations) {
        CUresult r;
        r = launch_kernel(ldpcHandle.get()->m_kernelNodeParams, m_cuStream);
        if(r != CUDA_SUCCESS) {
            throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
        }
    }
    else {

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float time = 0.0f;
        cudaEventRecord(start);

        // Launch the Kernel
        CUresult r;
        for(unsigned int i = 0; i < m_numIterations; i++)
        {
            r = launch_kernel(ldpcHandle.get()->m_kernelNodeParams, m_cuStream);
            if(r != CUDA_SUCCESS) {
                throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double tput = (K * C / 1000000) * (m_numIterations / time);
        std::cout << "Total time from C++ is " << time * 1000 << " us" << std::endl;
        std::cout << "Internal throughput is " << tput << " Gbps" << std::endl;
    }
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    ldpcHandle.reset();
    dLdpcDesc.reset();
    hLdpcDesc.reset();
    dWorkspace.reset();
    hWorkspace.reset();

    // Convert output from tensor device to host numpy
    uint32_t dim0 = dOutputTensor.dimensions()[0];
    uint32_t dim1 = dOutputTensor.dimensions()[1];

    cuphy::tensor_pinned hOutputTensor = cuphy::tensor_pinned(m_outputHostPtr, CUPHY_R_32F, dim0, dim1, cuphy::tensor_flags::align_tight);
    hOutputTensor.convert(dOutputTensor, m_cuStream);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Create the Numpy array for output.
    return hostToNumpy<float>((float*)m_outputHostPtr, effN, dim1);
}



} // namespace pycuphy
