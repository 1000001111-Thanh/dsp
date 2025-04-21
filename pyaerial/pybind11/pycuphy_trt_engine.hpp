/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_TRT_ENGINE_HPP
#define PYCUPHY_TRT_ENGINE_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"

namespace py = pybind11;


namespace pycuphy {

class TrtEngine {
public:
    TrtEngine(
        const std::string& trtModelFile,
        const uint32_t maxBatchSize,
        const std::vector<cuphyTrtTensorPrms_t>& inputTensorPrms,
        const std::vector<cuphyTrtTensorPrms_t>& outputTensorPrms,
        cudaStream_t cuStream);
    ~TrtEngine();

    // Use with fixed batch size which in this case is equal to maximum batch size.
    cuphyStatus_t run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors);

    // Use with dynamic batch size.
    cuphyStatus_t run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors, const uint32_t batchSize);

private:
    void init(const std::string& trtModelFile,
              const uint32_t maxBatchSize,
              const std::vector<cuphyTrtTensorPrms_t>& inputTensorPrms,
              const std::vector<cuphyTrtTensorPrms_t>& outputTensorPrms,
              cudaStream_t cuStream);

    void destroy();

    cuphyTrtEngineHndl_t m_trtEngineHndl;

    cudaStream_t m_cuStream;

    // Input and output tensors.
    std::vector<cuphyTrtTensorPrms_t> m_inputTensorPrms;
    std::vector<cuphyTrtTensorPrms_t> m_outputTensorPrms;
};


void getTensorPrms(const std::vector<std::string>& names,
                   const std::vector<std::vector<int>>& shapes,
                   const std::vector<cuphyDataType_t>& dataTypes,
                   std::vector<cuphyTrtTensorPrms_t>& tensorPrmsVec);


class __attribute__((visibility("default"))) PyTrtEngine {

public:
    PyTrtEngine(
        const std::string& trtModelFile,
        const uint32_t maxBatchSize,
        const std::vector<std::string>& inputNames,
        const std::vector<std::vector<int>>& inputShapes,
        const std::vector<cuphyDataType_t>& inputDataTypes,
        const std::vector<std::string>& outputNames,
        const std::vector<std::vector<int>>& outputShapes,
        const std::vector<cuphyDataType_t>& outputDataTypes,
        uint64_t cuStream);

    // Run using Python inputs/outputs.
    const std::vector<py::array>& run(const std::vector<py::array>& inputTensors);

private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;
    size_t getBufferSize(const uint32_t maxBatchSize,
                         const std::vector<std::vector<int>>& inputShapes,
                         const std::vector<std::vector<int>>& outputShapes) const;

    // Input and output tensors.
    uint8_t m_numInputs;
    std::vector<cuphyTrtTensorPrms_t> m_inputTensorPrms;
    std::vector<void*> m_inputBuffers;
    uint8_t m_numOutputs;
    std::vector<cuphyTrtTensorPrms_t> m_outputTensorPrms;
    std::vector<void*> m_outputBuffers;

    cudaStream_t m_cuStream;

    std::unique_ptr<TrtEngine> m_trtEngine;

    // Python outputs.
    std::vector<py::array> m_outputTensors;
};


} // pycuphy

#endif // PYCUPHY_TRT_ENGINE_HPP
