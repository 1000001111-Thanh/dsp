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

#include "cuphy.h"
#include "util.hpp"
#include "utils.cuh"
#include "cuphy_internal.h"
#include "pycuphy_util.hpp"
#include "pycuphy_ldpc.hpp"


namespace py = pybind11;

namespace pycuphy {


LdpcRateMatch::LdpcRateMatch(const uint64_t inputDevicePtr,
                             const uint64_t outputDevicePtr,
                             const uint64_t inputHostPtr,
                             const uint64_t tempOutputHostPtr,
                             const uint64_t outputHostPtr,
                             const bool scrambling,
                             const uint64_t cuStream):
m_inputDevicePtr((void *)inputDevicePtr),
m_outputDevicePtr((void *)outputDevicePtr),
m_inputHostPtr((void *)inputHostPtr),
m_tempOutputHostPtr((void *)tempOutputHostPtr),
m_outputHostPtr((float *)outputHostPtr),
m_scrambling(scrambling),
m_numIterations(0),
m_cuStream((cudaStream_t)cuStream)
{}


void LdpcRateMatch::setProfilingIterations(const uint32_t numIterations) {
    m_numIterations = numIterations;
}


py::array_t<float> LdpcRateMatch::rateMatch(const py::array& inputBits,
                                            const uint32_t tbSize,
                                            const float codeRate,
                                            const uint32_t rateMatchLen,
                                            const uint8_t qamMod,
                                            const uint8_t numLayers,
                                            const uint8_t rv,
                                            const uint32_t cinit) {

    const uint32_t ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits

    if (!m_inputDevicePtr || !m_outputDevicePtr || !m_outputHostPtr || !m_inputHostPtr || !m_tempOutputHostPtr) {
        throw std::runtime_error("LdpcRateMatch::rateMatch: Memory not allocated!");
    }

    // TODO: This is fixed and hard-coded for now.
    const int numTbs = 1;
    m_tbParams.resize(numTbs);

    // Access the input data address
    py::array_t<float, py::array::c_style | py::array::forcecast> array = inputBits;
    py::buffer_info buf = array.request();
    uint32_t N = buf.shape[0];
    uint32_t C = buf.shape[1];

    // Set transport block parameters.
    // TODO: Change this once multiple TBs get supported.
    setPdschPerTbParams(
        m_tbParams[0],
        tbSize,
        codeRate,
        rateMatchLen,
        qamMod,
        C,
        N,
        rv,
        numLayers,
        cinit
    );

    uint32_t maxN = 0, maxC = 0;
    for (int i = 0; i < numTbs; i++) {
        maxN = std::max(maxN, (uint32_t)m_tbParams[i].N);
        maxC = std::max(maxC, m_tbParams[i].num_CBs);
    }

    // Allocate workspace and copy config params
    size_t allocatedWorkspaceSize = cuphyDlRateMatchingWorkspaceSize(numTbs);
    cuphy::unique_device_ptr<uint32_t> dWorkspace = cuphy::make_unique_device<uint32_t>(div_round_up<uint32_t>(allocatedWorkspaceSize, sizeof(uint32_t)));
    cuphy::unique_pinned_ptr<uint32_t> hWorkspace = cuphy::make_unique_pinned<uint32_t>((2 + 2) * numTbs);


    // Copy TB parameters from host to device.
    cuphy::unique_device_ptr<PdschPerTbParams> dTbPrmsArray = cuphy::make_unique_device<PdschPerTbParams>(numTbs);
    CUDA_CHECK(cudaMemcpyAsync(dTbPrmsArray.get(), m_tbParams.data(), sizeof(PdschPerTbParams) * numTbs, cudaMemcpyHostToDevice, m_cuStream));

    // Device input tensor. Pre-allocated memory.
    uint32_t roundedN = round_up_to_next(N, ELEMENT_SIZE);
    cuphy::tensor_device dInputTensor = cuphy::tensor_device(m_inputDevicePtr, CUPHY_BIT, roundedN, C, numTbs);

    // Overprovisioned output buffers (device/host).
    const uint32_t maxEmax = PDSCH_MAX_ER_PER_CB_BITS;
    size_t maxNumOutputElems = div_round_up(numTbs * maxC * maxEmax, ELEMENT_SIZE);
    cuphy::tensor_device dOutputTensor = cuphy::tensor_device(m_outputDevicePtr,
                                                              CUPHY_R_32U,
                                                              maxNumOutputElems);

    cuphy::tensor_pinned hOutputTensor = cuphy::tensor_pinned(m_tempOutputHostPtr,
                                                              CUPHY_R_32U,
                                                              maxNumOutputElems);

    // Allocate launch config struct.
    std::unique_ptr<cuphyDlRateMatchingLaunchConfig> rmHandle = std::make_unique<cuphyDlRateMatchingLaunchConfig>();

    // Allocate descriptors and setup rate matching component
    uint8_t descAsyncCopy = 1; // Copy descriptor to the GPU during setup.

    size_t descSize = 0, allocSize = 0;
    cuphyStatus_t status = cuphyDlRateMatchingGetDescrInfo(&descSize, &allocSize);
    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("cuphyDlRateMatchingGetDescrInfo error!");
    }
    cuphy::unique_device_ptr<uint8_t> dRmDesc = cuphy::make_unique_device<uint8_t>(descSize);
    cuphy::unique_pinned_ptr<uint8_t> hRmDesc = cuphy::make_unique_pinned<uint8_t>(descSize);

    // Copy DMRS parameters from host to device.
    cuphy::unique_pinned_ptr<PdschDmrsParams> hDmrsParams = cuphy::make_unique_pinned<PdschDmrsParams>(numTbs);
    cuphy::unique_device_ptr<PdschDmrsParams> dDmrsParams = cuphy::make_unique_device<PdschDmrsParams>(numTbs);
    CUDA_CHECK(cudaMemcpyAsync(dDmrsParams.get(), hDmrsParams.get(), numTbs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice, m_cuStream));

    cuphyPdschStatusOut_t pdschStatusOut; // Populated during cuphySetupDlRateMatching, but contents not used here

    // Setup DL Rate Matching object
    const bool layerMapping = false;  // Hard-coded
    status = cuphySetupDlRateMatching(rmHandle.get(),
                                      &pdschStatusOut,
                                      (const uint32_t*)dInputTensor.addr(),
                                      (uint32_t*)dOutputTensor.addr(),
                                      nullptr,
                                      nullptr, // d_modulation_output
                                      nullptr, // d_xtf_re_map
                                      273,
                                      numTbs,
                                      0,
                                      m_scrambling,
                                      layerMapping,
                                      false,  // enable_modulation
                                      0,  // precoding
                                      false,  // restructure_kernel
                                      false,  // batching
                                      hWorkspace.get(),
                                      dWorkspace.get(),  // Explicit H2D copy as part of setup
                                      m_tbParams.data(),
                                      dTbPrmsArray.get(),
                                      dDmrsParams.get(),
                                      nullptr,  // d_ue_grp_params
                                      hRmDesc.get(),
                                      dRmDesc.get(),
                                      descAsyncCopy,
                                      m_cuStream);

    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cuphySetupDlRateMatching!");
    }

    uint32_t maxEr = 0;
    uint32_t* Er = (uint32_t*)hWorkspace.get() + 2 * numTbs;
    for (int i = 0; i < numTbs; i++) {
        uint32_t ErCb = (m_tbParams[i].testModel == 0) ? Er[i * 2 + 1] + (((m_tbParams[i].num_CBs - 1) < Er[i * 2]) ? 0 : m_tbParams[i].Nl * m_tbParams[i].Qm) : Er[i * 2 + 1];
        maxEr = std::max(maxEr, ErCb);
    }
    maxEr = div_round_up<uint32_t>(maxEr, ELEMENT_SIZE) * ELEMENT_SIZE;
    if (maxEr > maxEmax) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Emax {} but supported maximum Emax is {}", maxEr, maxEmax);
        throw std::runtime_error("Emax exceeds max supported!");
    }

    // Convert the input float array data to 32 bit array data. Memory is pre-allocated.
    // TODO: Once multiple TBs are supported, the size/type of these need to change.
    uint32_t divN = div_round_up(N, ELEMENT_SIZE);
    cuphy::tensor_pinned hInputTensor = cuphy::tensor_pinned(m_inputHostPtr, CUPHY_R_32U, divN, C, numTbs);
    fromNumpyBitArray<uint32_t>((float*)buf.ptr,
                                (uint32_t*)hInputTensor.addr(),
                                N,
                                C);
    CUDA_CHECK(cudaMemcpyAsync(dInputTensor.addr(),
                               hInputTensor.addr(),
                               dInputTensor.desc().get_size_in_bytes(),
                               cudaMemcpyHostToDevice,
                               m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Run the kernel.
    CUresult r;
    if(!m_numIterations) {
        r = launch_kernel(rmHandle.get()->m_kernelNodeParams[0], m_cuStream);
        if(r != CUDA_SUCCESS) {
            throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
        }
    }
    else {

        cudaEvent_t start, stop;
        float time = 0.0f;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        for (int iter = 0; iter < m_numIterations; iter++) {
            r = launch_kernel(rmHandle.get()->m_kernelNodeParams[0], m_cuStream);
            if(r != CUDA_SUCCESS) {
                throw std::runtime_error("LdpcEncoder::encode: Invalid argument for LDPC kernel launch!");
            }
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        double tput = ((double)rateMatchLen * (double)numTbs  / (double)1000) * ((double)m_numIterations / (double)time);
        std::cout << "Total time from C++ is " << time * 1000 << " us." << std::endl;
        std::cout << "Internal throughput is " << tput / 1000 << " Gbps." << std::endl;
    }

    // Copy to host memory.
    size_t numOutputElems = div_round_up(numTbs * maxC * maxEr, ELEMENT_SIZE);

    CUDA_CHECK(cudaMemcpyAsync(hOutputTensor.addr(),
                               dOutputTensor.addr(),
                               numOutputElems * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost,
                               m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Read output.
    int bitCount = 0;
    for(int tbIdx = 0; tbIdx < numTbs; tbIdx++) {

        for (int cbIdx = 0; cbIdx < m_tbParams[tbIdx].num_CBs; cbIdx++) {

            uint32_t ErCb = Er[tbIdx * 2 + 1] + ((cbIdx < Er[tbIdx * 2]) ? 0 : m_tbParams[tbIdx].Nl * m_tbParams[tbIdx].Qm);
            for (int bitIdx = 0; bitIdx < ErCb; bitIdx++) {

                int outIdx = tbIdx * maxC * maxEr + cbIdx * maxEr + bitIdx;
                int outWord = outIdx / ELEMENT_SIZE;
                int outBits = outIdx % ELEMENT_SIZE;

                uint32_t value = (*((uint32_t*)hOutputTensor.addr() + outWord) >> outBits) & 0x1;
                *(m_outputHostPtr + bitCount) = (float)value;
                bitCount += 1;
            }
        }
    }

    // Create the Numpy array for output.
    // TODO: Need to change the output format once numTbs > 1 gets supported.
    return hostToNumpy<float>((float*)m_outputHostPtr, bitCount, (uint32_t)numTbs);

}

}  // namespace pycuphy
