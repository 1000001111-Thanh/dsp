/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_PUSCH_HPP
#define PYCUPHY_PUSCH_HPP

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"
#include "hdf5hpp.hpp"
#include "pycuphy_params.hpp"

namespace py = pybind11;

namespace pycuphy {

class PuschPipeline {

public:

    PuschPipeline(const py::object& statPrms, uint64_t cuStream);
    ~PuschPipeline();

    void setupPuschRx(const py::object& dynPrms);
    void runPuschRx();
    void writeDbgBufSynch();

private:

    void createPuschRx(cudaStream_t cuStream);
    void destroyPuschRx();

    PuschParams                          m_puschParams;
    cuphyPuschRxHndl_t                   m_puschHandle;
    std::unique_ptr<hdf5hpp::hdf5_file>  m_debugFile;
};

} // namespace pycuphy

#endif // PYCUPHY_PUSCH_HPP