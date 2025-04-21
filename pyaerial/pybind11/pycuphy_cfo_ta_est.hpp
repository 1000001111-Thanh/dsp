/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_CFO_TA_EST_HPP
#define PYCUPHY_CFO_TA_EST_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"

namespace py = pybind11;

namespace pycuphy {

// This is the interface when called from C++.
class CfoTaEstimator {
public:

    CfoTaEstimator(cudaStream_t cuStream);
    ~CfoTaEstimator();

    // Run estimation. Note that the GPU memory pointers to the estimation results are
    // stored in puschParams, but can also be accessed using the getters below.
    void estimate(PuschParams& puschParams);

    // Output getters.
    const std::vector<cuphy::tensor_ref>& getCfoEst() const { return m_tCfoEstVec; }
    const cuphy::tensor_ref& getCfoHz() const { return m_tCfoHz; }
    const cuphy::tensor_ref& getTaEst() const { return m_tTaEst; }
    const cuphy::tensor_ref& getCfoPhaseRot() const { return m_tCfoPhaseRot; }
    const cuphy::tensor_ref& getTaPhaseRot() const { return m_tTaPhaseRot; }

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    cuphyPuschRxCfoTaEstHndl_t m_cfoTaEstHndl;

    // Descriptor variables.
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;
    size_t m_statDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_statDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_statDescrBufGpu;

    cudaStream_t m_cuStream;

    // Outputs.
    std::vector<cuphy::tensor_ref> m_tCfoEstVec;
    cuphy::tensor_ref m_tCfoHz;
    cuphy::tensor_ref m_tTaEst;
    cuphy::tensor_ref m_tCfoPhaseRot;
    cuphy::tensor_ref m_tTaPhaseRot;
    cuphy::tensor_ref m_tCfoTaEstInterCtaSyncCnt;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyCfoTaEstimator {
public:
    PyCfoTaEstimator(uint64_t cuStream);

    const std::vector<py::array_t<std::complex<float>>>& estimate(const std::vector<py::array_t<std::complex<float>>>& chEst,
                                                                  PuschParams& puschParams);

    const py::array_t<float>& getCfoHz() { return m_cfoEstHz; }
    const py::array_t<float>& getTaEst() { return m_taEst; }
    const py::array_t<std::complex<float>>& getCfoPhaseRot() { return m_cfoPhaseRot; }
    const py::array_t<std::complex<float>>& getTaPhaseRot() { return m_taPhaseRot; }

private:

    CfoTaEstimator m_cfoTaEstimator;

    cudaStream_t m_cuStream;

    // Inputs.
    std::vector<cuphy::tensor_device> m_tChannelEst;

    // Outputs.
    std::vector<py::array_t<std::complex<float>>> m_cfoEstVec;
    py::array_t<float> m_cfoEstHz;
    py::array_t<float> m_taEst;
    py::array_t<std::complex<float>> m_cfoPhaseRot;
    py::array_t<std::complex<float>> m_taPhaseRot;
};

} // pycuphy

#endif // PYCUPHY_CFO_TA_EST_HPP
