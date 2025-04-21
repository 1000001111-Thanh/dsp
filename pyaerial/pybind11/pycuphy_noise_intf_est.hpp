/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_NOISE_INTF_EST_HPP
#define PYCUPHY_NOISE_INTF_EST_HPP

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
class NoiseIntfEstimator {
public:

    NoiseIntfEstimator(cudaStream_t cuStream);
    ~NoiseIntfEstimator();

    // Run estimation. Note that the GPU memory pointers to the estimation results are
    // stored in puschParams, but can also be accessed using the getters below.
    void estimate(PuschParams& puschParams);

    // Output getters.
    const cuphy::tensor_ref& getInfoNoiseVarPreEq() const { return m_tInfoNoiseVarPreEq; };
    const cuphy::tensor_ref& getInfoNoiseIntfEstInterCtaSyncCnt() const { return m_tInfoNoiseIntfEstInterCtaSyncCnt; }
    const std::vector<cuphy::tensor_ref>& getInfoLwInv() const { return m_tInfoLwInv; }

    // For debugging purposes, dump noise variance estimates.
    void debugDump(H5DebugDump& debugDump, cudaStream_t cuStream = 0);

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    cuphyPuschRxNoiseIntfEstHndl_t m_puschRxNoiseIntfEstHndl;

    // Descriptor variables.
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    cudaStream_t m_cuStream;

    // Outputs.
    cuphy::tensor_ref m_tInfoNoiseVarPreEq;
    cuphy::tensor_ref m_tInfoNoiseIntfEstInterCtaSyncCnt;
    std::vector<cuphy::tensor_ref> m_tInfoLwInv;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyNoiseIntfEstimator {
public:
    PyNoiseIntfEstimator(uint64_t cuStream);

    const std::vector<py::array_t<std::complex<float>>>& estimate(const std::vector<py::array_t<std::complex<float>>>& chEst,
                                                                  PuschParams& puschParams);
    const std::vector<float>& getInvNoiseVarLin() { return m_invNoiseVarLin; }
    const py::array_t<float>& getInfoNoiseVarPreEq() { return m_infoNoiseVarPreEq; }

private:

    NoiseIntfEstimator m_noiseIntfEstimator;

    cudaStream_t m_cuStream;

    // Inputs.
    std::vector<cuphy::tensor_device> m_tChannelEst;

    // Outputs.
    py::array_t<float> m_infoNoiseVarPreEq;
    std::vector<py::array_t<std::complex<float>>> m_infoLwInv;
    std::vector<float> m_invNoiseVarLin;
};

} // pycuphy

#endif // PYCUPHY_NOISE_INTF_EST_HPP