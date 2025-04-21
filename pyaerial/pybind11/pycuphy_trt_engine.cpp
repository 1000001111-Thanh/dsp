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
#include <numeric>
#include <functional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "tensor_desc.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_trt_engine.hpp"

namespace py = pybind11;

namespace pycuphy {

TrtEngine::TrtEngine(const std::string& trtModelFile,
                     const uint32_t maxBatchSize,
                     const std::vector<cuphyTrtTensorPrms_t>& inputTensorPrms,
                     const std::vector<cuphyTrtTensorPrms_t>& outputTensorPrms,
                     cudaStream_t cuStream) {
    init(trtModelFile, maxBatchSize, inputTensorPrms, outputTensorPrms, cuStream);
}


void TrtEngine::init(const std::string& trtModelFile,
                     const uint32_t maxBatchSize,
                     const std::vector<cuphyTrtTensorPrms_t>& inputTensorPrms,
                     const std::vector<cuphyTrtTensorPrms_t>& outputTensorPrms,
                     cudaStream_t cuStream) {
    m_inputTensorPrms = inputTensorPrms;
    m_outputTensorPrms = outputTensorPrms;
    m_cuStream = cuStream;

    cuphyStatus_t status = cuphyCreateTrtEngine(&m_trtEngineHndl,
                                                trtModelFile.c_str(),
                                                maxBatchSize,
                                                m_inputTensorPrms.data(),
                                                m_inputTensorPrms.size(),
                                                m_outputTensorPrms.data(),
                                                m_outputTensorPrms.size(),
                                                m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateTrtEngine()");
    }

}


TrtEngine::~TrtEngine() {
    destroy();
}


void TrtEngine::destroy() {
    cuphyStatus_t status = cuphyDestroyTrtEngine(m_trtEngineHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyTrtEngine()");
    }
}


cuphyStatus_t TrtEngine::run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors) {
    return run(inputTensors, outputTensors, 0);
}


cuphyStatus_t TrtEngine::run(const std::vector<void*>& inputTensors, const std::vector<void*>& outputTensors, const uint32_t batchSize) {
    cuphyStatus_t setupStatus = cuphySetupTrtEngine(m_trtEngineHndl,
                                                    (void**)inputTensors.data(),
                                                    inputTensors.size(),
                                                    (void**)outputTensors.data(),
                                                    outputTensors.size(),
                                                    batchSize);
    if(setupStatus != CUPHY_STATUS_SUCCESS) {
        return setupStatus;
    }

    cuphyStatus_t runStatus = cuphyRunTrtEngine(m_trtEngineHndl, m_cuStream);
    return runStatus;
}


PyTrtEngine::PyTrtEngine(const std::string& trtModelFile,
                         const uint32_t maxBatchSize,
                         const std::vector<std::string>& inputNames,
                         const std::vector<std::vector<int>>& inputShapes,
                         const std::vector<cuphyDataType_t>& inputDataTypes,
                         const std::vector<std::string>& outputNames,
                         const std::vector<std::vector<int>>& outputShapes,
                         const std::vector<cuphyDataType_t>& outputDataTypes,
                         uint64_t cuStream):
m_linearAlloc(getBufferSize(maxBatchSize, inputShapes, outputShapes)),
m_numInputs(inputNames.size()),
m_numOutputs(outputNames.size()),
m_cuStream((cudaStream_t)cuStream) {

    getTensorPrms(inputNames, inputShapes, inputDataTypes, m_inputTensorPrms);
    getTensorPrms(outputNames, outputShapes, outputDataTypes, m_outputTensorPrms);

    m_trtEngine = std::make_unique<TrtEngine>(trtModelFile, maxBatchSize, m_inputTensorPrms, m_outputTensorPrms, m_cuStream);

    // Allocate buffers for inputs and outputs
    m_inputBuffers.resize(m_numInputs);
    for(int i = 0; i < m_numInputs; i++)  {
        size_t nBytes = get_cuphy_type_storage_element_size(inputDataTypes[i]);
        nBytes *= std::accumulate(inputShapes[i].cbegin(), inputShapes[i].cend(), 1, std::multiplies<int>{});
        nBytes *= maxBatchSize;
        m_inputBuffers[i] = m_linearAlloc.alloc(nBytes);
    }

    // Output tensors.
    m_outputBuffers.resize(m_numOutputs);
    for(int i = 0; i < m_numOutputs; i++)  {
        size_t nBytes = get_cuphy_type_storage_element_size(outputDataTypes[i]);
        nBytes *= std::accumulate(outputShapes[i].cbegin(), outputShapes[i].cend(), 1, std::multiplies<int>{});
        nBytes *= maxBatchSize;
        m_outputBuffers[i] = m_linearAlloc.alloc(nBytes);

        // Add batch size as the first dimension.
        for(int j = m_outputTensorPrms[i].nDims; j > 0; j--) {
            m_outputTensorPrms[i].dims[j] = m_outputTensorPrms[i].dims[j-1];
        }
        m_outputTensorPrms[i].dims[0] = maxBatchSize;
        m_outputTensorPrms[i].nDims += 1;
    }

    m_outputTensors.resize(m_numOutputs);
}


const std::vector<py::array>& PyTrtEngine::run(const std::vector<py::array>& inputTensors) {

    if(m_numInputs != inputTensors.size()) {
        throw std::runtime_error("Invalid number TRT inputs!");
    }

    // Move input tensors from host to pre-allocated device memory.
    uint32_t batchSize;
    for(int index = 0; index < m_numInputs; index++) {

        switch(m_inputTensorPrms[index].dataType) {
        case CUPHY_R_32F:
            {
                deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(py::cast<py::array_t<float>>(inputTensors[index]),
                                                                                  m_inputBuffers[index],
                                                                                  m_inputTensorPrms[index].dataType,
                                                                                  m_inputTensorPrms[index].dataType,
                                                                                  cuphy::tensor_flags::align_tight,
                                                                                  m_cuStream);
                // Read actual batch size.
                py::array_t<float, py::array::f_style | py::array::forcecast> inputTensor = inputTensors[index];
                py::buffer_info info = inputTensor.request();
                batchSize = info.shape[0];
                break;
            }
        case CUPHY_R_32I:
            {
                deviceFromNumpy<int, py::array::f_style | py::array::forcecast>(py::cast<py::array_t<int>>(inputTensors[index]),
                                                                                m_inputBuffers[index],
                                                                                m_inputTensorPrms[index].dataType,
                                                                                m_inputTensorPrms[index].dataType,
                                                                                cuphy::tensor_flags::align_tight,
                                                                                m_cuStream);
                // Read actual batch size.
                py::array_t<int, py::array::f_style | py::array::forcecast> inputTensor = inputTensors[index];
                py::buffer_info info = inputTensor.request();
                batchSize = info.shape[0];
                break;
            }
        default:
            throw std::runtime_error("Invalid input data type!");
            break;
        }
    }

    cuphyStatus_t status = m_trtEngine->run(m_inputBuffers, m_outputBuffers, batchSize);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyRunTrtEngine()");
    }

    // Move outputs from device to host memory and to Python arrays.
    for(int i = 0; i < m_numOutputs; i++) {

        // Set batch size correctly.
        m_outputTensorPrms[i].dims[0] = batchSize;

        // Copy to host.
        cuphy::tensor_layout tLayout = cuphy::tensor_layout(m_outputTensorPrms[i].nDims, m_outputTensorPrms[i].dims, 0);
        cuphy::tensor_info tInfo = cuphy::tensor_info(m_outputTensorPrms[i].dataType, tLayout);
        cuphy::tensor_device dTensor = cuphy::tensor_device(m_outputBuffers[i], tInfo);
        cuphy::tensor_pinned hTensor = cuphy::tensor_pinned(tInfo);

        hTensor.convert(dTensor, m_cuStream);
        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

        // Create the Numpy array.
        std::vector<size_t> strides;  // Default ones used if this is empty.
        std::vector<size_t> dims = std::vector<size_t>(m_outputTensorPrms[i].dims, m_outputTensorPrms[i].dims + m_outputTensorPrms[i].nDims);

        switch(m_outputTensorPrms[i].dataType) {
        case CUPHY_R_32F:
            m_outputTensors[i] = hostToNumpy<float>((float*)hTensor.addr(), dims, strides);
            break;
        case CUPHY_R_32I:
            m_outputTensors[i] = hostToNumpy<int>((int*)hTensor.addr(), dims, strides);
            break;
        default:
            throw std::runtime_error("Invalid output data type!");
            break;
        }
    }

    return m_outputTensors;
}


size_t PyTrtEngine::getBufferSize(const uint32_t maxBatchSize,
                                  const std::vector<std::vector<int>>& inputShapes,
                                  const std::vector<std::vector<int>>& outputShapes) const {

    // Allocate internal buffers.
    const uint32_t N_BYTES_PER_ELEM = 4;
    const uint32_t EXTRA_PADDING = LINEAR_ALLOC_PAD_BYTES;

    size_t nBytesBuffer = 0;
    for(const auto& shape : inputShapes) {
        nBytesBuffer += N_BYTES_PER_ELEM * maxBatchSize * std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>{}) + EXTRA_PADDING;
    }
    for(const auto& shape : outputShapes) {
        nBytesBuffer += N_BYTES_PER_ELEM * maxBatchSize * std::accumulate(shape.cbegin(), shape.cend(), 1, std::multiplies<int>{}) + EXTRA_PADDING;
    }

    return nBytesBuffer;
}


void getTensorPrms(const std::vector<std::string>& names,
                   const std::vector<std::vector<int>>& shapes,
                   const std::vector<cuphyDataType_t>& dataTypes,
                   std::vector<cuphyTrtTensorPrms_t>& tensorPrmsVec) {
    int numTensors = names.size();
    tensorPrmsVec.resize(numTensors);
    for(int i = 0; i < numTensors; i++)  {
        cuphyTrtTensorPrms_t tensorPrms;
        tensorPrms.name = names[i].c_str();
        tensorPrms.nDims = shapes[i].size();
        tensorPrms.dataType = dataTypes[i];
        memcpy(tensorPrms.dims, shapes[i].data(), tensorPrms.nDims * sizeof(int));
        tensorPrmsVec[i] = tensorPrms;
    }
}


} // namespace pycuphy
