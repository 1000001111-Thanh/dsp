/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_LDPC_HPP
#define PYCUPHY_LDPC_HPP

#include <vector>
#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"
#include "pusch_utils.hpp"
#include "pycuphy_params.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


namespace pycuphy {

//////////////////////////////////////////////////////////////////////////////
// Common utility functions
void printPdschPerTbParams(const PdschPerTbParams& tbParams);

void printPerTbParams(const PerTbParams& tbParams);

void setPdschPerTbParams(PdschPerTbParams& tbParams,
                         const uint32_t tbSize,
                         const float codeRate,
                         const uint32_t rateMatchLen,
                         const uint8_t qamMod,
                         const uint32_t numCodeBlocks,
                         const uint32_t numCodedBits,
                         const uint8_t rv,
                         const uint8_t numLayers,
                         const uint32_t cinit);

void setPerTbParams(PerTbParams& tbParams,
                    cuphyLDPCParams& ldpcParams,
                    const uint32_t tbSize,
                    const float codeRate,
                    const uint8_t qamMod,
                    const uint32_t ndi,
                    const uint32_t rv,
                    const uint32_t rateMatchLen,
                    const uint32_t cinit,
                    const uint32_t userGroupIdx,
                    const uint8_t numLayers,
                    const uint8_t numUeGrpLayers,
                    const std::vector<uint32_t>& layerMapArray,
                    const uint8_t nDmrsCdmGrpsNoData = 2);


//////////////////////////////////////////////////////////////////////////////
// LDPC encoder wrapper

class LdpcEncoder {

public:
    LdpcEncoder(
        const uint64_t inputDevicePtr,
        const uint64_t tempInputHostPtr,
        const uint64_t outputDevicePtr,
        const uint64_t outputHostPtr,
        const uint64_t cuStream
    );

    py::array_t<float> encode(
        const py::array& inputData,
        const uint32_t tbSize,
        const float codeRate,
        const int rv
    );

    void setProfilingIterations(const uint16_t numIterations);
    void setPuncturing(const uint8_t puncture);

private:

    void *                          m_inputDevicePtr;       // Input device (GPU) memory pointer
    void *                          m_outputDevicePtr;      // Output device (GPU) memory pointer
    void *                          m_outputHostPtr;        // Output host (CPU) memory pointer
    void *                          m_tempInputHostPtr;     // Temporary input host pointer

    uint8_t                         m_puncture;             // Puncturing flag
    uint16_t                        m_numIterations;        // Number of profiling iterations

    std::vector<PdschPerTbParams>   m_tbParams;             // Transport block parameters

    cudaStream_t m_cuStream;                                // CUDA stream

};


//////////////////////////////////////////////////////////////////////////////
// LDPC rate matching wrapper

class LdpcRateMatch {
    public:

        LdpcRateMatch(const uint64_t inputDevicePtr,
                      const uint64_t outputDevicePtr,
                      const uint64_t inputHostPtr,
                      const uint64_t tempOutputHostPtr,
                      const uint64_t outputHostPtr,
                      const bool scrambling,
                      const uint64_t cuStream);

        py::array_t<float> rateMatch(const py::array& inputBits,
                                     const uint32_t tbSize,
                                     const float codeRate,
                                     const uint32_t rateMatchLen,
                                     const uint8_t qamMod,
                                     const uint8_t numLayers,
                                     const uint8_t rv,
                                     const uint32_t cinit);

        void setProfilingIterations(const uint32_t numIterations);

    private:

        void *                m_inputDevicePtr;         // Input device (GPU) memory pointer
        void *                m_outputDevicePtr;        // Output device (GPU) memory pointer
        void *                m_inputHostPtr;           // Input host (CPU) memory pointer
        void *                m_tempOutputHostPtr;      // Temporary output host (CPU) memory pointer
        float *               m_outputHostPtr;          // Output host (CPU) memory pointer
        cudaStream_t          m_cuStream;               // CUDA stream

        std::vector<PdschPerTbParams>      m_tbParams;  // Transport block parameters

        bool                  m_scrambling;             // Enable/disable scrambling
        uint32_t              m_numIterations;          // Number of profiling iterations
};


//////////////////////////////////////////////////////////////////////////////
// LDPC de-rate match wrappers

// This is the C++ API.
class LdpcDerateMatch {
public:
    LdpcDerateMatch(const bool scrambling, const cudaStream_t cuStream);
    ~LdpcDerateMatch();

    void derateMatch(const std::vector<cuphy::tensor_ref>& llrs, void** deRmOutput, PuschParams& puschParams);

    void derateMatch(const std::vector<cuphyTensorPrm_t>& inputLlrs,
                     void** deRmOutput,
                     const PerTbParams* pTbPrmsCpu,
                     const PerTbParams* pTbPrmsGpu,
                     int nUes);

private:

    void destroy();

    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    cuphyPuschRxRateMatchHndl_t m_puschRmHndl;
    cudaStream_t                m_cuStream;
};

// This is the Python API exposed to Python through pybind11.
class __attribute__((visibility("default"))) PyLdpcDerateMatch {
public:
    PyLdpcDerateMatch(const bool scrambling, const uint64_t cuStream);
    ~PyLdpcDerateMatch();

    const std::vector<py::array_t<float>>& derateMatch(const std::vector<py::array_t<float>>& inputLlrs,
                                                       const std::vector<uint32_t>& tbSizes,
                                                       const std::vector<float>& codeRates,
                                                       const std::vector<uint32_t>& rateMatchLengths,
                                                       const std::vector<uint8_t>& qamMods,
                                                       const std::vector<uint8_t>& numLayers,
                                                       const std::vector<uint32_t>& rvs,
                                                       const std::vector<uint32_t>& ndis,
                                                       const std::vector<uint32_t>& cinits,
                                                       const std::vector<uint32_t>& userGroupIdxs);

private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    size_t getBufferSize() const;

    LdpcDerateMatch m_derateMatch;
    cudaStream_t m_cuStream;

    std::vector<cuphy::tensor_device> m_inputLlrTensors;

    // Internal buffer.
    void** m_deRmOutput;

    // Python output.
    std::vector<py::array_t<float>> m_pyDeRmOutput;
};


//////////////////////////////////////////////////////////////////////////////
// LDPC decoder wrappers

// This is the C++ API.
class LdpcDecoder {

public:

    LdpcDecoder(const cudaStream_t cuStream);

    void* decode(void** deRmOutput, PuschParams& puschParams);

    void* decode(const std::vector<void*>& deRmLlr,
                 const std::vector<PerTbParams>& tbPrmsCpu,
                 const cuphyLDPCParams& ldpcParams);

    const std::vector<void*>& getSoftOutputs() const;

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;
    size_t getBufferSize() const;

    // Output address on device.
    void* m_ldpcOutput;

    // Output address of soft outputs on device.
    std::vector<void*> m_ldpcSoftOutput;

    cuphy::context      m_ctx;
    cuphy::LDPC_decoder m_decoder;
    cudaStream_t        m_cuStream;

    // Normalization factor for min-sum.
    float               m_normalizationFactor;

};

// This is the Python API exposed to Python through pybind11.
class __attribute__((visibility("default"))) PyLdpcDecoder {

public:
    PyLdpcDecoder(const uint64_t cuStream);

    const std::vector<py::array_t<float>>& decode(
        const std::vector<py::array_t<float>>& inputLlrs,
        const std::vector<uint32_t>& tbSizes,
        const std::vector<float>& codeRates,
        const std::vector<uint32_t>& rvs,
        const std::vector<uint32_t>& rateMatchLengths
    );

    void setNumIterations(const uint32_t numIterations);
    void setThroughputMode(const uint8_t throughputMode);

    const std::vector<py::array_t<float>>& getSoftOutputs();

private:
    LdpcDecoder m_decoder;
    cudaStream_t m_cuStream;

    std::vector<cuphy::tensor_device> m_inputLlrTensors;
    cuphyLDPCParams m_ldpcParams;
    std::vector<PerTbParams> m_tbParams;

    std::vector<py::array_t<float>> m_ldpcOutput;
    std::vector<py::array_t<float>> m_softOutput;

};



}

#endif // PYCUPHY_LDPC_HPP