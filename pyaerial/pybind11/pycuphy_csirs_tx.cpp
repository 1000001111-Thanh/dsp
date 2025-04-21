/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <vector>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "pycuphy_csirs_tx.hpp"
#include "pycuphy_util.hpp"

namespace py = pybind11;

namespace pycuphy {


void printCsiRsDynPrms(const cuphyCsirsDynPrms_t& dynPrms) {
    std::cout << "\ncuphyCsirsDynPrms_t:" << std::endl;
    std::cout << "====================" << std::endl;

    std::cout << "cuStream: " << (uint64_t)dynPrms.cuStream << std::endl;
    std::cout << "nCells: " << dynPrms.nCells << std::endl;
    std::cout << "procModeBmsk: " << dynPrms.procModeBmsk << std::endl;
    std::cout << "nPrecodingMatrices: " << dynPrms.nPrecodingMatrices << std::endl;

    int nRrcPrms = 0;
    for(int cellIdx = 0; cellIdx < dynPrms.nCells; cellIdx++) {
        const cuphyCsirsCellDynPrm_t& cellDynPrms = dynPrms.pCellParam[cellIdx];
        std::cout << "Cell " << cellIdx << ": cuphyCsirsCellDynPrm_t : " << std::endl;
        std::cout << "===============================" << std::endl;

        std::cout << "rrcParamsOffset: " << cellDynPrms.rrcParamsOffset << std::endl;
        std::cout << "nRrcParams: " << (int)cellDynPrms.nRrcParams << std::endl;
        std::cout << "slotBufferIdx: " << cellDynPrms.slotBufferIdx << std::endl;
        std::cout << "cellPrmStatIdx: " << cellDynPrms.cellPrmStatIdx << std::endl;
        nRrcPrms += cellDynPrms.nRrcParams;
    }
    for(int rrcPrmIdx = 0; rrcPrmIdx < nRrcPrms; rrcPrmIdx++) {
        std::cout << "CSI-RS RRC cuphyCsirsRrcDynPrm_t: " << rrcPrmIdx << std::endl;
        std::cout << "======================================" << std::endl;
        const cuphyCsirsRrcDynPrm_t& rrcDynPrms = dynPrms.pRrcDynPrm[rrcPrmIdx];
        printCsiRsRrcDynPrms(rrcDynPrms);
    }
}


void printCsiRsRrcDynPrms(const cuphyCsirsRrcDynPrm_t& rrcDynPrms) {
    std::cout << "startRb: " << rrcDynPrms.startRb << std::endl;
    std::cout << "nRb: " << rrcDynPrms.nRb << std::endl;
    std::cout << "freqDomain: " << rrcDynPrms.freqDomain << std::endl;
    std::cout << "row: " << (int)rrcDynPrms.row << std::endl;
    std::cout << "symbL0: " << (int)rrcDynPrms.symbL0 << std::endl;
    std::cout << "symbL1: " << (int)rrcDynPrms.symbL1 << std::endl;
    std::cout << "freqDensity: " << (int)rrcDynPrms.freqDensity << std::endl;
    std::cout << "scrambId: " << rrcDynPrms.scrambId << std::endl;
    std::cout << "idxSlotInFrame: " << (int)rrcDynPrms.idxSlotInFrame << std::endl;
    std::cout << "csiType: " << rrcDynPrms.csiType << std::endl;
    std::cout << "cdmType: " << rrcDynPrms.cdmType << std::endl;
    std::cout << "beta: " << rrcDynPrms.beta << std::endl;
    std::cout << "enablePrcdBf: " << (int)rrcDynPrms.enablePrcdBf << std::endl;
    std::cout << "pmwPrmIdx: " << rrcDynPrms.pmwPrmIdx << std::endl;
}


void readCsiRsRrcDynPrms(const py::object& pyCsiRsRrcDynPrms, cuphyCsirsRrcDynPrm_t& csiRsRrcDynPrms) {
    csiRsRrcDynPrms.startRb = pyCsiRsRrcDynPrms.attr("start_prb").cast<uint16_t>();
    csiRsRrcDynPrms.nRb = pyCsiRsRrcDynPrms.attr("num_prb").cast<uint16_t>();
    const py::list prbBitmap = pyCsiRsRrcDynPrms.attr("prb_bitmap");
    uint16_t freqDomain = 0;
    for(int i = 0; i < 16; i++) {
        const uint16_t bit = prbBitmap[prbBitmap.size() - i - 1].cast<uint16_t>();
        freqDomain |= ((bit & 0x001) << i);
    }
    csiRsRrcDynPrms.freqDomain = freqDomain;
    csiRsRrcDynPrms.row = pyCsiRsRrcDynPrms.attr("row").cast<uint8_t>();
    csiRsRrcDynPrms.symbL0 = pyCsiRsRrcDynPrms.attr("symb_L0").cast<uint8_t>();
    csiRsRrcDynPrms.symbL1 = pyCsiRsRrcDynPrms.attr("symb_L1").cast<uint8_t>();
    csiRsRrcDynPrms.freqDensity = pyCsiRsRrcDynPrms.attr("freq_density").cast<uint8_t>();
    csiRsRrcDynPrms.scrambId = pyCsiRsRrcDynPrms.attr("scramb_id").cast<uint16_t>();
    csiRsRrcDynPrms.idxSlotInFrame = pyCsiRsRrcDynPrms.attr("idx_slot_in_frame").cast<uint8_t>();
    csiRsRrcDynPrms.csiType = (cuphyCsiType_t)pyCsiRsRrcDynPrms.attr("csi_type").cast<uint8_t>();
    csiRsRrcDynPrms.cdmType = (cuphyCdmType_t)pyCsiRsRrcDynPrms.attr("cdm_type").cast<uint8_t>();
    csiRsRrcDynPrms.beta = pyCsiRsRrcDynPrms.attr("beta").cast<float>();
    csiRsRrcDynPrms.enablePrcdBf = pyCsiRsRrcDynPrms.attr("enable_precoding").cast<uint8_t>();
    csiRsRrcDynPrms.pmwPrmIdx = pyCsiRsRrcDynPrms.attr("pmw_prm_idx").cast<uint16_t>();
}


CsiRsTx::CsiRsTx(cuphyCsirsStatPrms_t const* pStatPrms) {
    cuphyStatus_t status = cuphyCreateCsirsTx(&m_csirsTxHndl, pStatPrms);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateCsirsTx()");
    }
}


cuphyStatus_t CsiRsTx::run(cuphyCsirsDynPrms_t* pDynPrms) {
    cuphyStatus_t status = cuphySetupCsirsTx(m_csirsTxHndl, pDynPrms);
    if(status != CUPHY_STATUS_SUCCESS) {
        return status;
    }

    status = cuphyRunCsirsTx(m_csirsTxHndl);
    return status;
}


CsiRsTx::~CsiRsTx() {
    cuphyStatus_t status = cuphyDestroyCsirsTx(m_csirsTxHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        std::cerr << "Failed to destroy CSI-RS Tx (cuphyDestroyCsirsTx)!" << "\n";
    }
}


PyCsiRsTx::PyCsiRsTx(const std::vector<uint16_t>& numPrbDlBwp) {
    uint16_t numCells = numPrbDlBwp.size();

    m_tracker.pMemoryFootprint = nullptr;

    m_cellStatPrms.resize(numCells);
    for(int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        memset(&m_cellStatPrms[cellIdx], 0, sizeof(cuphyCellStatPrm_t));  // Other parameters not used, set to zero.
        m_cellStatPrms[cellIdx].nPrbDlBwp = numPrbDlBwp[cellIdx];
    }

    m_statPrms.pOutInfo = &m_tracker;
    m_statPrms.pCellStatPrms = m_cellStatPrms.data();
    m_statPrms.nCells = numCells;
    m_statPrms.nMaxCellsPerSlot = numCells;

    m_csiRsTx = std::make_unique<CsiRsTx>(&m_statPrms);
}


const std::vector<py::array_t<std::complex<float>>>& PyCsiRsTx::run(const py::list& pyCsiRsCellDynPrms, const py::list& precodingMatrices, const std::vector<py::array_t<std::complex<float>>>& txBuffers, uint64_t cudaStream) {

    // Create the dynamic parameters object.
    m_dynPrms.cuStream = (cudaStream_t)cudaStream;

    const uint16_t numCells = pyCsiRsCellDynPrms.size();
    m_dynPrms.nCells = numCells;

    uint16_t nTotalRrcPrms = 0;
    m_csiRsCellDynPrms.resize(numCells);
    for(int i = 0; i < numCells; i++) {
        const py::list rrcDynPrmsList = pyCsiRsCellDynPrms[i].attr("rrc_dyn_prms");
        const uint16_t nRrcPrms = rrcDynPrmsList.size();

        m_csiRsCellDynPrms[i].rrcParamsOffset = nTotalRrcPrms;
        m_csiRsCellDynPrms[i].nRrcParams = nRrcPrms;
        m_csiRsCellDynPrms[i].slotBufferIdx = i;
        m_csiRsCellDynPrms[i].cellPrmStatIdx = i;

        nTotalRrcPrms += nRrcPrms;
    }
    m_dynPrms.pCellParam = m_csiRsCellDynPrms.data();

    m_csiRsRrcDynPrms.resize(nTotalRrcPrms);
    int count = 0;
    for(int i = 0; i < numCells; i++) {
        const py::list rrcDynPrmsList = pyCsiRsCellDynPrms[i].attr("rrc_dyn_prms");
        for(int j = 0; j < rrcDynPrmsList.size(); j++) {
            readCsiRsRrcDynPrms(rrcDynPrmsList[j], m_csiRsRrcDynPrms[count]);
            count++;
        }
    }
    m_dynPrms.pRrcDynPrm = m_csiRsRrcDynPrms.data();

    m_dynPrms.procModeBmsk = CSIRS_PROC_MODE_GRAPHS;
    m_dynPrms.chan_graph = nullptr;

    // Data output.
    m_txBuffer.resize(numCells);
    m_txBufferTensorPrm.resize(numCells);
    if(txBuffers.size() != numCells) {
        throw std::runtime_error("The number of Tx buffers does not match with CSI-RS cell dynamic parameters!");
    }

    for (int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        py::array_t<std::complex<float>> cellTxBuffer = txBuffers[cellIdx];
        m_txBuffer[cellIdx] = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
            cellTxBuffer,
            CUPHY_C_32F,
            CUPHY_C_16F,
            cuphy::tensor_flags::align_tight,
            m_dynPrms.cuStream);
        m_txBufferTensorPrm[cellIdx].desc = m_txBuffer[cellIdx].desc().handle();
        m_txBufferTensorPrm[cellIdx].pAddr = m_txBuffer[cellIdx].addr();
    }

    m_csiRsDataOut.pTDataTx = m_txBufferTensorPrm.data();
    m_dynPrms.pDataOut = &m_csiRsDataOut;

    // Precoding matrices.
    m_dynPrms.nPrecodingMatrices = precodingMatrices.size();
    m_csiRsPmW.resize(m_dynPrms.nPrecodingMatrices);
    if(m_dynPrms.nPrecodingMatrices > 0) {
        for(int pmwIdx = 0; pmwIdx < m_dynPrms.nPrecodingMatrices; pmwIdx++) {
            const py::object precodingMatrix = precodingMatrices[pmwIdx];
            m_csiRsPmW[pmwIdx].nPorts = precodingMatrix.attr("num_ports").cast<uint8_t>();

            py::array temp = precodingMatrix.attr("precoding_matrix");
            py::array_t<std::complex<float>> pmwArray = temp;
            py::buffer_info buf = pmwArray.request();
            std::complex<float> *ptr = static_cast<std::complex<float> *>(buf.ptr);
            for (size_t idx = 0; idx < buf.size; idx++){
                m_csiRsPmW[pmwIdx].matrix[idx].x = __float2half(ptr[idx].real());
                m_csiRsPmW[pmwIdx].matrix[idx].y = __float2half(ptr[idx].imag());
            }
        }
        m_dynPrms.pPmwParams = m_csiRsPmW.data();
    }
    else
        m_dynPrms.pPmwParams = nullptr;

    // Run CSI-RS transmission.
    cuphyStatus_t status = m_csiRsTx->run(&m_dynPrms);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to run CSI-RS Tx (CsiRxTx::run)!");
    }

    // Device to host copy for the Tx buffer.
    m_txBufferOut.resize(numCells);
    for(int cellIdx = 0; cellIdx < numCells; cellIdx++) {

        cuphy::tensor_pinned txBufferHost = cuphy::tensor_pinned(CUPHY_C_32F,
                                                                 m_txBuffer[cellIdx].layout(),
                                                                 cuphy::tensor_flags::align_tight);
        txBufferHost.convert(m_txBuffer[cellIdx], m_dynPrms.cuStream);
        CUDA_CHECK(cudaStreamSynchronize(m_dynPrms.cuStream));

        std::vector<size_t> strides;  // Default strides used if this is empty.
        std::vector<size_t> dims = std::vector<size_t>(m_txBuffer[cellIdx].dimensions().begin(), m_txBuffer[cellIdx].dimensions().begin() + m_txBuffer[cellIdx].rank());
        m_txBufferOut[cellIdx] = hostToNumpy<std::complex<float>>((std::complex<float>*)txBufferHost.addr(), dims, strides);
    }
    return m_txBufferOut;
}



}  // namespace pycuphy