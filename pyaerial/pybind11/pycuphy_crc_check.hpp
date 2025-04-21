/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_CRC_CHECK_HPP
#define PYCUPHY_CRC_CHECK_HPP

#include <vector>
#include "cuphy.h"
#include "util.hpp"
#include "cuphy.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


namespace pycuphy {


// This is the C++ API.
class CrcChecker {

public:
    CrcChecker(const cudaStream_t cuStream);
    ~CrcChecker();

    void checkCrc(void* ldpcOutput,
                  PuschParams& puschparams);

    void checkCrc(void* ldpcOutput,
                  const PerTbParams* tbPrmsCpu,
                  const PerTbParams* tbPrmsGpu,
                  const int nUes);

    const uint8_t* getOutputTbs() const { return m_outputTbs; }
    const uint32_t* getCbCrcs() const { return m_outputCbCrcs; }
    const uint32_t* getTbCrcs() const { return m_outputTbCrcs; }

    const std::vector<uint32_t>& getTbPayloadStartOffsets() const { return m_tbPayloadStartOffsets; }
    const std::vector<uint32_t>& getTbCrcStartOffsets() const { return m_tbCrcStartOffsets; }
    const std::vector<uint32_t>& getCbCrcStartOffsets() const { return m_cbCrcStartOffsets; }

    uint32_t getTotNumTbs() const { return m_totNumTbs; }
    uint32_t getTotNumCbs() const { return m_totNumCbs; }
    uint32_t getTotNumPayloadBytes() const { return m_totNumPayloadBytes; }

private:
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void allocateDescr();
    size_t getBufferSize() const;
    void destroy();

    // Descriptor variables.
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    // Output addresses on device.
    uint8_t* m_outputTbs;
    uint32_t* m_outputCbCrcs;
    uint32_t* m_outputTbCrcs;

    // Start offsets per UE.
    std::vector<uint32_t> m_tbPayloadStartOffsets;
    std::vector<uint32_t> m_tbCrcStartOffsets;
    std::vector<uint32_t> m_cbCrcStartOffsets;

    // Total number of TBs/CBs/payload bytes.
    uint32_t m_totNumTbs;
    uint32_t m_totNumCbs;
    uint32_t m_totNumPayloadBytes;

    cudaStream_t m_cuStream;
    cuphyPuschRxCrcDecodeHndl_t m_crcDecodeHndl;
};


// This is the Python API exposed to Python through pybind11.
class __attribute__((visibility("default"))) PyCrcChecker {

public:
    PyCrcChecker(const uint64_t cuStream);

    const std::vector<py::array_t<uint8_t>>& checkCrc(const py::array_t<float>& ldpcOutput,
                                                      const std::vector<uint32_t>& tbSizes,
                                                      const std::vector<float>& codeRates);

    const std::vector<py::array_t<uint8_t>>& getTbPayloads() const { return m_tbPayloads; }
    const std::vector<py::array_t<uint32_t>>& getCbCrcs() const { return m_cbCrcs; }
    const std::vector<py::array_t<uint32_t>>& getTbCrcs() const { return m_tbCrcs; }

private:
    CrcChecker m_crcChecker;
    cudaStream_t m_cuStream;

    std::vector<py::array_t<uint8_t>> m_tbPayloads;
    std::vector<py::array_t<uint32_t>> m_tbCrcs;
    std::vector<py::array_t<uint32_t>> m_cbCrcs;
};


} // namespace pycuphy


#endif // PYCUPHY_CRC_CHECK_HPP