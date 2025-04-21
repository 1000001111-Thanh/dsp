/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_CHANNEL_EST_HPP
#define PYCUPHY_CHANNEL_EST_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "pycuphy_debug.hpp"
#include "pycuphy_params.hpp"
#include "IModule.hpp"

namespace py = pybind11;


namespace pycuphy {


// This is the interface when called from C++.
class ChannelEstimator {
public:

    enum DescriptorTypes {
        CH_EST               = 0,
        N_CH_EST_DESCR_TYPES = CH_EST + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST + 1
    };

    ChannelEstimator(const PuschParams& puschParams, cudaStream_t cuStream);
    ~ChannelEstimator();

    // Run estimation. Note that the GPU memory pointers to the estimation results are
    // stored in puschParams, but can also be accessed using the getters below.
    void estimate(PuschParams& puschParams);

    // Full channel estimate. Note that this does not get populated if only LS estimates are requested.
    const std::vector<cuphy::tensor_ref>& getChEst() const { return m_tChannelEst; };

    // LS channel estimates. Note that this only gets populated if using the LS+MMSE multi-stage algorithm,
    // or LS channel estimation only.
    const std::vector<cuphy::tensor_ref>& getLsChEst() const { return m_tDmrsLSEst; }

    const std::vector<cuphy::tensor_ref>& getDmrsDelayMean() const { return m_tDmrsDelayMean; }

    // For debugging purposes, dump channel estimates.
    void debugDump(H5DebugDump& debugDump, uint16_t numUeGrps, cudaStream_t cuStream = 0);

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void init(const PuschParams& puschParams);
    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    // Descriptor variables.
    cuphy::kernelDescrs<N_CH_EST_DESCR_TYPES> m_kernelStatDescr;
    cuphy::kernelDescrs<N_CH_EST_DESCR_TYPES> m_kernelDynDescr;

    cudaStream_t m_cuStream;

    // Outputs.
    std::vector<cuphy::tensor_ref> m_tChannelEst, m_tDbg, m_tDmrsLSEst, m_tDmrsDelayMean, m_tDmrsAccum;

    std::unique_ptr<cuphy::IKernelBuilder> m_chestKernelBuilder;
    std::unique_ptr<cuphy::IModule> m_chest;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyChannelEstimator {
public:
    PyChannelEstimator(const PuschParams& puschParams, uint64_t cuStream);

    const std::vector<py::array_t<std::complex<float>>>& estimate(PuschParams& puschParams);

private:
    ChannelEstimator m_chEstimator;
    cudaStream_t m_cuStream;

    // Outputs.
    std::vector<py::array_t<std::complex<float>>> m_chEst;
};


} // pycuphy

#endif // PYCUPHY_CHANNEL_EST_HPP
