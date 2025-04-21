/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_PDSCH_HPP
#define PYCUPHY_PDSCH_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"
#include "pycuphy_params.hpp"

namespace py = pybind11;

namespace pycuphy {

class PdschPipeline {

public:
    PdschPipeline(const py::object& statPrms);
    ~PdschPipeline();

    void setupPdschTx(const py::object& dynPrms);
    void runPdschTx();
    py::array_t<float> getLdpcOutputPerTbPerCell(int cellIdx, int tbIdx, const uint64_t ldpcOutputHostPtr);

private:
    PdschParams m_pdschParams;
    cuphyPdschTxHndl_t m_pdschHandle;

    void createPdschTx();
    void destroyPdschTx();
};


}  // namespace pycuphy

#endif