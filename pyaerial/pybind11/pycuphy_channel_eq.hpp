/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_CHANNEL_EQ_HPP
#define PYCUPHY_CHANNEL_EQ_HPP

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
class ChannelEqualizer {
public:
    enum DescriptorTypes {
        PUSCH_CH_EQ_COEF             = 0,
        PUSCH_CH_EQ_SOFT_DEMAP       = PUSCH_CH_EQ_COEF + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ,
        PUSCH_CH_EQ_IDFT             = PUSCH_CH_EQ_SOFT_DEMAP + 1,
        N_CH_EQ_DESCR_TYPES          = PUSCH_CH_EQ_IDFT + 1
    };


    ChannelEqualizer(cudaStream_t cuStream);
    ~ChannelEqualizer();

    // Run equalization. Note that the GPU memory pointers to the equalization results are
    // stored in puschParams, but can also be accessed using the getters below.
    // This assumes that also inputs are already in puschParams (which is the case if
    // channel estimation, noise/interference estimation have been called first).
    void equalize(PuschParams& puschParams);

    // Access the results once the equalizer has been run.
    const std::vector<cuphy::tensor_ref>& getChEqCoef() const { return m_tCoef; };
    const std::vector<cuphy::tensor_ref>& getReeDiagInv() const { return m_tReeDiagInv; };
    const std::vector<cuphy::tensor_ref>& getDataEq() const { return m_tDataEq; };
    const std::vector<cuphy::tensor_ref>& getLlr() const { return m_tLLR; };
    const std::vector<cuphy::tensor_ref>& getLlrCdm1() const { return m_tLLRCdm1; };

private:

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_linearAlloc;

    void destroy();
    void allocateDescr();
    size_t getBufferSize() const;

    cuphyPuschRxChEqHndl_t m_chEqHndl;
    cuphy::context m_ctx;

    // Descriptor variables.
    cuphy::kernelDescrs<N_CH_EQ_DESCR_TYPES> m_kernelStatDescr;
    cuphy::kernelDescrs<N_CH_EQ_DESCR_TYPES> m_kernelDynDescr;

    cudaStream_t m_cuStream;

    // Outputs on device.
    std::vector<cuphy::tensor_ref> m_tCoef, m_tReeDiagInv, m_tDbg, m_tDataEq, m_tLLR, m_tLLRCdm1;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PyChannelEqualizer {
public:
    PyChannelEqualizer(uint64_t cuStream);

    const std::vector<py::array_t<float>>& equalize(const std::vector<py::array_t<std::complex<float>>>& chEst,
                                                    const std::vector<py::array_t<std::complex<float>>>& infoLwInv,
                                                    const py::array& infoNoiseVarPreEq,
                                                    const py::array& invNoiseVarLin,
                                                    PuschParams& puschParams);

    // Others via getters.
    const std::vector<py::array_t<std::complex<float>>>& getDataEq() { return m_dataEq; }
    const std::vector<py::array_t<std::complex<float>>>& getEqCoef() { return m_coef; }
    const std::vector<py::array_t<float>>& getReeDiagInv() { return m_ReeDiagInv; }

private:
    ChannelEqualizer m_chEqualizer;
    cudaStream_t m_cuStream;

    // Inputs.
    std::vector<cuphy::tensor_device> m_tChannelEst;
    std::vector<cuphy::tensor_device> m_tInfoLwInv;
    cuphy::tensor_device m_tInfoNoiseVarPreEq;

    // Outputs to Python.
    std::vector<py::array_t<std::complex<float>>> m_coef;
    std::vector<py::array_t<float>> m_ReeDiagInv;
    std::vector<py::array_t<std::complex<float>>> m_dataEq;
    std::vector<py::array_t<float>> m_LLR;
};


} // pycuphy

#endif // PYCUPHY_CHANNEL_EQ_HPP
