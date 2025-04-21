/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_RSRP_HPP
#define PYCUPHY_RSRP_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "pycuphy_util.hpp"
#include "pycuphy_debug.hpp"
#include "pycuphy_params.hpp"

namespace py = pybind11;

namespace pycuphy {

// This is the interface when called from C++.
class RsrpEstimator {
public:

    RsrpEstimator(cudaStream_t cuStream);
    ~RsrpEstimator();

    // Run estimation. Note that the GPU memory pointers to the estimation results are
    // stored in puschParams, but can also be accessed using the getters below.
    void estimate(PuschParams& puschParams);

    // Access the results once the estimator has been run.
    const cuphy::tensor_ref& getRsrp() const { return m_tRsrp; }
    const cuphy::tensor_ref& getNoiseIntVarPostEq() const { return m_tNoiseIntfVarPostEq; }
    const cuphy::tensor_ref& getSinrPreEq() const { return m_tSinrPreEq; }
    const cuphy::tensor_ref& getSinrPostEq() const { return m_tSinrPostEq; }

    void debugDump(H5DebugDump& debugDump, cudaStream_t cuStream = 0);

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    cuphyPuschRxRssiHndl_t m_puschRxRssiHndl;

    // Descriptor variables.
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    cudaStream_t m_cuStream;

    // Outputs.
    cuphy::tensor_ref m_tRsrp;
    cuphy::tensor_ref m_tNoiseIntfVarPostEq;
    cuphy::tensor_ref m_tSinrPreEq;
    cuphy::tensor_ref m_tSinrPostEq;
    cuphy::tensor_ref m_tInterCtaSyncCnt;

};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyRsrpEstimator {
public:
    PyRsrpEstimator(uint64_t cuStream);

    const py::array_t<float>& estimate(const std::vector<py::array_t<std::complex<float>>>& chEst,
                                       const std::vector<py::array_t<float>>& reeDiagInv,
                                       const py::array& infoNoiseVarPreEq,
                                       PuschParams& puschParams);

    // output getters to be used after the estimation has been run.
    const py::array_t<float>& getInfoNoiseVarPostEq() { return m_infoNoiseVarPostEq; }
    const py::array_t<float>& getSinrPreEq() const { return m_sinrPreEq; }
    const py::array_t<float>& getSinrPostEq() const { return m_sinrPostEq; }

private:

    RsrpEstimator m_rsrpEstimator;

    cudaStream_t m_cuStream;

    // Inputs.
    std::vector<cuphy::tensor_device> m_tChannelEst;
    std::vector<cuphy::tensor_device> m_tReeDiagInv;
    cuphy::tensor_device m_tInfoNoiseVarPreEq;

    // Outputs.
    py::array_t<float> m_rsrp;
    py::array_t<float> m_infoNoiseVarPostEq;
    py::array_t<float> m_sinrPreEq;
    py::array_t<float> m_sinrPostEq;

};

} // pycuphy

#endif // PYCUPHY_RSRP_HPP