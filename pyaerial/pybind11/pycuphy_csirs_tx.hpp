/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_CSIRS_TX_HPP
#define PYCUPHY_CSIRS_TX_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"

namespace py = pybind11;


namespace pycuphy {

void printCsiRsDynPrms(const cuphyCsirsDynPrms_t& dynPrms);
void printCsiRsRrcDynPrms(const cuphyCsirsRrcDynPrm_t& rrcDynPrms);
void readCsiRsRrcDynPrms(const py::object& pyCsiRsRrcDynPrms, cuphyCsirsRrcDynPrm_t& csiRsRrcDynPrms);


// This is the C++ wrapper around cuPHY.
class CsiRsTx {

public:
    CsiRsTx(cuphyCsirsStatPrms_t const* pStatPrms);
    ~CsiRsTx();
    cuphyStatus_t run(cuphyCsirsDynPrms_t* pDynPrms);

private:
    cuphyCsirsTxHndl_t m_csirsTxHndl;
};


// This is for the Python bindings.
class PyCsiRsTx {

public:
    PyCsiRsTx(const std::vector<uint16_t>& numPrbDlBwp);

    const std::vector<py::array_t<std::complex<float>>>& run(const py::list& pyCsiRsCellDynPrms,
                                                             const py::list& precodingMatrices,
                                                             const std::vector<py::array_t<std::complex<float>>>& txBuffers,
                                                             uint64_t cudaStream);

private:
    cuphyCsirsStatPrms_t    m_statPrms;
    cuphyCsirsDynPrms_t     m_dynPrms;

    cuphyTracker_t                      m_tracker;
    std::vector<cuphyCellStatPrm_t>     m_cellStatPrms;
    std::vector<cuphyCsirsRrcDynPrm_t>  m_csiRsRrcDynPrms;
    std::vector<cuphyCsirsCellDynPrm_t> m_csiRsCellDynPrms;
    cuphyCsirsDataOut_t                 m_csiRsDataOut;
    std::vector<cuphyPmWOneLayer_t>     m_csiRsPmW;

    // Inputs.
    std::vector<cuphyTensorPrm_t>       m_txBufferTensorPrm;
    std::vector<cuphy::tensor_device>   m_txBuffer;

    // Outputs.
    std::vector<py::array_t<std::complex<float>>> m_txBufferOut;

    std::unique_ptr<CsiRsTx> m_csiRsTx;
};


}  // namespace pycuphy


#endif // PYCUPHY_CSIRS_TX_HPP