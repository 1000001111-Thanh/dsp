/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "pycuphy_util.hpp"
#include "tensor_desc.hpp"
#include "pycuphy_srs_chest.hpp"

namespace py = pybind11;

namespace pycuphy {

constexpr uint16_t SRS_BW_TABLE[64][8] =
   {{4,1,4,1,4,1,4,1},
    {8,1,4,2,4,1,4,1},
    {12,1,4,3,4,1,4,1},
    {16,1,4,4,4,1,4,1},
    {16,1,8,2,4,2,4,1},
    {20,1,4,5,4,1,4,1},
    {24,1,4,6,4,1,4,1},
    {24,1,12,2,4,3,4,1},
    {28,1,4,7,4,1,4,1},
    {32,1,16,2,8,2,4,2},
    {36,1,12,3,4,3,4,1},
    {40,1,20,2,4,5,4,1},
    {48,1,16,3,8,2,4,2},
    {48,1,24,2,12,2,4,3},
    {52,1,4,13,4,1,4,1},
    {56,1,28,2,4,7,4,1},
    {60,1,20,3,4,5,4,1},
    {64,1,32,2,16,2,4,4},
    {72,1,24,3,12,2,4,3},
    {72,1,36,2,12,3,4,3},
    {76,1,4,19,4,1,4,1},
    {80,1,40,2,20,2,4,5},
    {88,1,44,2,4,11,4,1},
    {96,1,32,3,16,2,4,4},
    {96,1,48,2,24,2,4,6},
    {104,1,52,2,4,13,4,1},
    {112,1,56,2,28,2,4,7},
    {120,1,60,2,20,3,4,5},
    {120,1,40,3,8,5,4,2},
    {120,1,24,5,12,2,4,3},
    {128,1,64,2,32,2,4,8},
    {128,1,64,2,16,4,4,4},
    {128,1,16,8,8,2,4,2},
    {132,1,44,3,4,11,4,1},
    {136,1,68,2,4,17,4,1},
    {144,1,72,2,36,2,4,9},
    {144,1,48,3,24,2,12,2},
    {144,1,48,3,16,3,4,4},
    {144,1,16,9,8,2,4,2},
    {152,1,76,2,4,19,4,1},
    {160,1,80,2,40,2,4,10},
    {160,1,80,2,20,4,4,5},
    {160,1,32,5,16,2,4,4},
    {168,1,84,2,28,3,4,7},
    {176,1,88,2,44,2,4,11},
    {184,1,92,2,4,23,4,1},
    {192,1,96,2,48,2,4,12},
    {192,1,96,2,24,4,4,6},
    {192,1,64,3,16,4,4,4},
    {192,1,24,8,8,3,4,2},
    {208,1,104,2,52,2,4,13},
    {216,1,108,2,36,3,4,9},
    {224,1,112,2,56,2,4,14},
    {240,1,120,2,60,2,4,15},
    {240,1,80,3,20,4,4,5},
    {240,1,48,5,16,3,8,2},
    {240,1,24,10,12,2,4,3},
    {256,1,128,2,64,2,4,16},
    {256,1,128,2,32,4,4,8},
    {256,1,16,16,8,2,4,2},
    {264,1,132,2,44,3,4,11},
    {272,1,136,2,68,2,4,17},
    {272,1,68,4,4,17,4,1},
    {272,1,16,17,8,2,4,2}};


PySrsChannelEstimator::PySrsChannelEstimator(const py::dict& chEstParams, uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_srsChEstimator((cudaStream_t)cuStream),
m_nPrbs(273) {
    readChEstParams(chEstParams);
    m_srsChEstimator.init(m_srsFilterPrms);
}


const std::vector<py::array_t<std::complex<float>>>& PySrsChannelEstimator::estimate(
        const py::array& inputData,
        const uint16_t nSrsUes,
        const uint16_t nCells,
        const uint16_t nPrbGrps,
        const uint16_t startPrbGrp,
        const std::vector<py::object>& pySrsCellPrms,
        const std::vector<py::object>& pyUeSrsPrms) {

    m_nSrsUes = nSrsUes;

    m_srsRbSnrBuffer.resize(nSrsUes * m_nPrbs);
    m_srsReports.resize(nSrsUes);

    m_chEsts.clear();

    // Convert Python structs into cuPHY.
    std::vector<cuphySrsCellPrms_t> srsCellPrms;
    readSrsCellParams(srsCellPrms, pySrsCellPrms);
    std::vector<cuphyUeSrsPrm_t> ueSrsPrms;
    readUeSrsParams(ueSrsPrms, pyUeSrsPrms);

    // Read input data into device memory.
    std::vector<cuphyTensorPrm_t> tDataRx(nCells);
    cuphy::tensor_device deviceRxDataTensor = deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
        inputData,
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);

    // Only one cell supported through the Python API for now.
    tDataRx[0].desc = deviceRxDataTensor.desc().handle();
    tDataRx[0].pAddr = deviceRxDataTensor.addr();

    std::vector<cuphySrsChEstBuffInfo_t> srsChEstBuffInfo = m_srsChEstimator.estimate(tDataRx,
                                                                                      nSrsUes,
                                                                                      nCells,
                                                                                      nPrbGrps,
                                                                                      startPrbGrp,
                                                                                      srsCellPrms,
                                                                                      ueSrsPrms);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Create the return value.
    for(int ueIdx=0; ueIdx < nSrsUes; ueIdx++) {
        cuphyTensorPrm_t* pChEst = &srsChEstBuffInfo[ueIdx].tChEstBuffer;
        const ::tensor_desc& tDesc = static_cast<const ::tensor_desc&>(*pChEst->desc);
        const ::tensor_layout_any& tLayout = tDesc.layout();

        uint16_t nPrbGrpEsts = tLayout.dimensions[0];
        uint16_t nGnbAnts = tLayout.dimensions[1];
        uint16_t nUeAnts = tLayout.dimensions[2];

        // Copy results to host buffers.
        cuphy::tensor_device dChEstTensor = cuphy::tensor_device((std::complex<float>*)pChEst->pAddr, CUPHY_C_32F, nPrbGrpEsts, nGnbAnts, nUeAnts, cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hChEstTensor = cuphy::tensor_pinned(CUPHY_C_32F, nPrbGrpEsts, nGnbAnts, nUeAnts, cuphy::tensor_flags::align_tight);
        hChEstTensor.convert(dChEstTensor, m_cuStream);
        CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

        // Create the numpy array for the output channel estimates.
        py::array_t<std::complex<float>> chEst = hostToNumpy<std::complex<float>>((std::complex<float>*)hChEstTensor.addr(),
                                                                                  nPrbGrpEsts, nGnbAnts, nUeAnts);

        // The return value is a vector of Numpy arrays.
        m_chEsts.push_back(chEst);
    }

    cuphySrsReport_t* pSrsReports = m_srsChEstimator.getSrsReport();
    CUDA_CHECK(cudaMemcpyAsync(m_srsReports.data(),
                               pSrsReports,
                               sizeof(cuphySrsReport_t) * nSrsUes,
                               cudaMemcpyDeviceToHost,
                               m_cuStream));

    float* pRbSnrBuffer = m_srsChEstimator.getRbSnrBuffer();
    uint32_t rbSnrBufferSize = nSrsUes * m_nPrbs;
    CUDA_CHECK(cudaMemcpyAsync(m_srsRbSnrBuffer.data(),
                               pRbSnrBuffer,
                               sizeof(float) * rbSnrBufferSize,
                               cudaMemcpyDeviceToHost,
                               m_cuStream));

    return m_chEsts;
}


void PySrsChannelEstimator::readSrsCellParams(std::vector<cuphySrsCellPrms_t>& srsCellPrms, const std::vector<py::object>& pySrsCellPrms) {
    srsCellPrms.resize(pySrsCellPrms.size());
    for(int cellIdx = 0; cellIdx < srsCellPrms.size(); cellIdx++)
    {
        srsCellPrms[cellIdx].slotNum        = pySrsCellPrms[cellIdx].attr("slot_num").cast<uint16_t>();
        srsCellPrms[cellIdx].frameNum       = pySrsCellPrms[cellIdx].attr("frame_num").cast<uint16_t>();
        srsCellPrms[cellIdx].srsStartSym    = pySrsCellPrms[cellIdx].attr("srs_start_sym").cast<uint8_t>();
        srsCellPrms[cellIdx].nSrsSym        = pySrsCellPrms[cellIdx].attr("num_srs_sym").cast<uint8_t>();
        srsCellPrms[cellIdx].nRxAntSrs      = pySrsCellPrms[cellIdx].attr("num_rx_ant_srs").cast<uint16_t>();
        srsCellPrms[cellIdx].mu             = pySrsCellPrms[cellIdx].attr("mu").cast<uint8_t>();
    }
}


void PySrsChannelEstimator::readUeSrsParams(std::vector<cuphyUeSrsPrm_t>& ueSrsPrms, const std::vector<py::object>& pyUeSrsPrms) {
    ueSrsPrms.resize(pyUeSrsPrms.size());
    for(int ueIdx = 0; ueIdx < pyUeSrsPrms.size(); ueIdx++)
    {
        ueSrsPrms[ueIdx].cellIdx = pyUeSrsPrms[ueIdx].attr("cell_idx").cast<uint16_t>();
        ueSrsPrms[ueIdx].nAntPorts = pyUeSrsPrms[ueIdx].attr("num_ant_ports").cast<uint8_t>();
        ueSrsPrms[ueIdx].nSyms = pyUeSrsPrms[ueIdx].attr("num_syms").cast<uint8_t>();
        ueSrsPrms[ueIdx].nRepetitions = pyUeSrsPrms[ueIdx].attr("num_repetitions").cast<uint8_t>();
        ueSrsPrms[ueIdx].combSize = pyUeSrsPrms[ueIdx].attr("comb_size").cast<uint8_t>();
        ueSrsPrms[ueIdx].startSym = pyUeSrsPrms[ueIdx].attr("start_sym").cast<uint8_t>();
        ueSrsPrms[ueIdx].sequenceId = pyUeSrsPrms[ueIdx].attr("sequence_id").cast<uint16_t>();
        ueSrsPrms[ueIdx].configIdx = pyUeSrsPrms[ueIdx].attr("config_idx").cast<uint8_t>();
        ueSrsPrms[ueIdx].bandwidthIdx = pyUeSrsPrms[ueIdx].attr("bandwidth_idx").cast<uint8_t>();
        ueSrsPrms[ueIdx].combOffset = pyUeSrsPrms[ueIdx].attr("comb_offset").cast<uint8_t>();
        ueSrsPrms[ueIdx].cyclicShift = pyUeSrsPrms[ueIdx].attr("cyclic_shift").cast<uint8_t>();
        ueSrsPrms[ueIdx].frequencyPosition = pyUeSrsPrms[ueIdx].attr("frequency_position").cast<uint8_t>();
        ueSrsPrms[ueIdx].frequencyShift = pyUeSrsPrms[ueIdx].attr("frequency_shift").cast<uint16_t>();
        ueSrsPrms[ueIdx].frequencyHopping = pyUeSrsPrms[ueIdx].attr("frequency_hopping").cast<uint8_t>();
        ueSrsPrms[ueIdx].resourceType = pyUeSrsPrms[ueIdx].attr("resource_type").cast<uint8_t>();
        ueSrsPrms[ueIdx].Tsrs = pyUeSrsPrms[ueIdx].attr("periodicity").cast<uint16_t>();
        ueSrsPrms[ueIdx].Toffset = pyUeSrsPrms[ueIdx].attr("offset").cast<uint16_t>();
        ueSrsPrms[ueIdx].groupOrSequenceHopping = pyUeSrsPrms[ueIdx].attr("group_or_sequence_hopping").cast<uint8_t>();
        ueSrsPrms[ueIdx].chEstBuffIdx = pyUeSrsPrms[ueIdx].attr("ch_est_buff_idx").cast<uint16_t>();
        py::array srs_ant_port_to_ue_ant_map = pyUeSrsPrms[ueIdx].attr("srs_ant_port_to_ue_ant_map");
        py::buffer_info buf = srs_ant_port_to_ue_ant_map.request();
        memcpy(&ueSrsPrms[ueIdx].srsAntPortToUeAntMap, buf.ptr, 4);
        ueSrsPrms[ueIdx].rnti = 0;
        ueSrsPrms[ueIdx].handle = 0;
        ueSrsPrms[ueIdx].prgSize = pyUeSrsPrms[ueIdx].attr("prg_size").cast<uint8_t>();
        ueSrsPrms[ueIdx].usage = 0;
    }
}


void PySrsChannelEstimator::readChEstParams(const py::dict& chEstParams) {
    m_tPrmFocc_table = deviceFromNumpy<std::complex<float>>(
        chEstParams["focc_table"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmFocc_table.desc = m_tPrmFocc_table.desc().handle();
    m_srsFilterPrms.tPrmFocc_table.pAddr = m_tPrmFocc_table.addr();

    m_tPrmFocc_comb2_table = deviceFromNumpy<std::complex<float>>(
        chEstParams["focc_table_comb2"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmFocc_comb2_table.desc  = m_tPrmFocc_comb2_table.desc().handle();
    m_srsFilterPrms.tPrmFocc_comb2_table.pAddr = m_tPrmFocc_comb2_table.addr();

    m_tPrmFocc_comb4_table = deviceFromNumpy<std::complex<float>>(
        chEstParams["focc_table_comb4"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmFocc_comb4_table.desc  = m_tPrmFocc_comb4_table.desc().handle();
    m_srsFilterPrms.tPrmFocc_comb4_table.pAddr = m_tPrmFocc_comb4_table.addr();

    m_tPrmW_comb2_nPorts1_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts1_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts1_wide.desc = m_tPrmW_comb2_nPorts1_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts1_wide.pAddr = m_tPrmW_comb2_nPorts1_wide.addr();

    m_tPrmW_comb2_nPorts2_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts2_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts2_wide.desc = m_tPrmW_comb2_nPorts2_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts2_wide.pAddr = m_tPrmW_comb2_nPorts2_wide.addr();

    m_tPrmW_comb2_nPorts4_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts4_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts4_wide.desc = m_tPrmW_comb2_nPorts4_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts4_wide.pAddr = m_tPrmW_comb2_nPorts4_wide.addr();

    m_tPrmW_comb2_nPorts8_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts8_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts8_wide.desc = m_tPrmW_comb2_nPorts8_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts8_wide.pAddr = m_tPrmW_comb2_nPorts8_wide.addr();


    m_tPrmW_comb4_nPorts1_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts1_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts1_wide.desc = m_tPrmW_comb4_nPorts1_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts1_wide.pAddr = m_tPrmW_comb4_nPorts1_wide.addr();

    m_tPrmW_comb4_nPorts2_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts2_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts2_wide.desc = m_tPrmW_comb4_nPorts2_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts2_wide.pAddr = m_tPrmW_comb4_nPorts2_wide.addr();

    m_tPrmW_comb4_nPorts4_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts4_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts4_wide.desc = m_tPrmW_comb4_nPorts4_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts4_wide.pAddr = m_tPrmW_comb4_nPorts4_wide.addr();

    m_tPrmW_comb4_nPorts6_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts6_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts6_wide.desc = m_tPrmW_comb4_nPorts6_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts6_wide.pAddr = m_tPrmW_comb4_nPorts6_wide.addr();

    m_tPrmW_comb4_nPorts12_wide = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts12_wide"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts12_wide.desc = m_tPrmW_comb4_nPorts12_wide.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts12_wide.pAddr = m_tPrmW_comb4_nPorts12_wide.addr();

    m_tPrmW_comb2_nPorts1_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts1_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts1_narrow.desc = m_tPrmW_comb2_nPorts1_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts1_narrow.pAddr = m_tPrmW_comb2_nPorts1_narrow.addr();

    m_tPrmW_comb2_nPorts2_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts2_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts2_narrow.desc = m_tPrmW_comb2_nPorts2_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts2_narrow.pAddr = m_tPrmW_comb2_nPorts2_narrow.addr();

    m_tPrmW_comb2_nPorts4_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts4_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts4_narrow.desc = m_tPrmW_comb2_nPorts4_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts4_narrow.pAddr = m_tPrmW_comb2_nPorts4_narrow.addr();

    m_tPrmW_comb2_nPorts8_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb2_nPorts8_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb2_nPorts8_narrow.desc = m_tPrmW_comb2_nPorts8_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb2_nPorts8_narrow.pAddr = m_tPrmW_comb2_nPorts8_narrow.addr();

    m_tPrmW_comb4_nPorts1_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts1_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts1_narrow.desc = m_tPrmW_comb4_nPorts1_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts1_narrow.pAddr = m_tPrmW_comb4_nPorts1_narrow.addr();

    m_tPrmW_comb4_nPorts2_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts2_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts2_narrow.desc = m_tPrmW_comb4_nPorts2_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts2_narrow.pAddr = m_tPrmW_comb4_nPorts2_narrow.addr();

    m_tPrmW_comb4_nPorts4_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts4_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts4_narrow.desc = m_tPrmW_comb4_nPorts4_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts4_narrow.pAddr = m_tPrmW_comb4_nPorts4_narrow.addr();

    m_tPrmW_comb4_nPorts6_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts6_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts6_narrow.desc = m_tPrmW_comb4_nPorts6_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts6_narrow.pAddr = m_tPrmW_comb4_nPorts6_narrow.addr();

    m_tPrmW_comb4_nPorts12_narrow = deviceFromNumpy<std::complex<float>>(
        chEstParams["W_comb4_nPorts12_narrow"],
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
    m_srsFilterPrms.tPrmW_comb4_nPorts12_narrow.desc = m_tPrmW_comb4_nPorts12_narrow.desc().handle();
    m_srsFilterPrms.tPrmW_comb4_nPorts12_narrow.pAddr = m_tPrmW_comb4_nPorts12_narrow.addr();

    m_srsFilterPrms.noisEstDebias_comb2_nPorts1 = chEstParams["noisEstDebias_comb2_nPorts1"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts2 = chEstParams["noisEstDebias_comb2_nPorts2"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts4 = chEstParams["noisEstDebias_comb2_nPorts4"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb2_nPorts8 = chEstParams["noisEstDebias_comb2_nPorts8"].cast<float>();

    m_srsFilterPrms.noisEstDebias_comb4_nPorts1  = chEstParams["noisEstDebias_comb4_nPorts1"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts2  = chEstParams["noisEstDebias_comb4_nPorts2"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts4  = chEstParams["noisEstDebias_comb4_nPorts4"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts6  = chEstParams["noisEstDebias_comb4_nPorts6"].cast<float>();
    m_srsFilterPrms.noisEstDebias_comb4_nPorts12 = chEstParams["noisEstDebias_comb4_nPorts12"].cast<float>();
}


py::array_t<float> PySrsChannelEstimator::getRbSnrBuffer() const {
    py::array_t<float> rbSnrs = hostToNumpy<float>((float*)m_srsRbSnrBuffer.data(), m_nPrbs, m_nSrsUes);
    return rbSnrs;
}


size_t SrsChannelEstimator::getBufferSize() const {
    const size_t maxNumSrsUes        = CUPHY_SRS_MAX_N_USERS;
    const int32_t EXTRA_PADDING      = maxNumSrsUes * 128; // Upper bound for extra memory required per allocation due to 128 alignment

    const size_t maxCells            = 1;  // TODO: Only one cell currently.
    const size_t maxRbSnrMem         = maxNumSrsUes * m_nPrbs * sizeof(float) + EXTRA_PADDING;
    const size_t maxSrsReportMem     = maxNumSrsUes * sizeof(cuphySrsReport_t) + EXTRA_PADDING;

    const size_t maxChEstToL2Mem     = maxCells * m_nPrbs * MAX_N_ANTENNAS_SUPPORTED * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL * sizeof(float2) * CUPHY_SRS_MAX_FULL_BAND_CHEST_PER_TTI + EXTRA_PADDING;
    size_t nBytesBuffer = maxRbSnrMem + maxSrsReportMem + maxChEstToL2Mem;
    return nBytesBuffer;
}


void SrsChannelEstimator::allocateDescr() {
    size_t statDescrAlignBytes, dynDescrAlignBytes;
    cuphyStatus_t status = cuphySrsChEstGetDescrInfo(&m_statDescrSizeBytes,
                                                     &statDescrAlignBytes,
                                                     &m_dynDescrSizeBytes,
                                                     &dynDescrAlignBytes);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphySrsChEstGetDescrInfo()");
    }

    m_statDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_statDescrSizeBytes);
    m_statDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_statDescrSizeBytes);
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);
}


SrsChannelEstimator::SrsChannelEstimator(cudaStream_t cuStream):
m_cuStream(cuStream),
m_nPrbs(273),  // Use maximum.
m_linearAlloc(getBufferSize()) {}


SrsChannelEstimator::SrsChannelEstimator(const cuphySrsFilterPrms_t& srsFilterPrms, cudaStream_t cuStream):
m_cuStream(cuStream),
m_nPrbs(273),  // Use maximum.
m_linearAlloc(getBufferSize()) {
    init(srsFilterPrms);
}


void SrsChannelEstimator::init(const cuphySrsFilterPrms_t& srsFilterPrms) {

    m_srsFilterPrms = srsFilterPrms;

    m_linearAlloc.memset(0, m_cuStream);
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));

    // Allocate descriptors.
    allocateDescr();

    // Create the SRS channel estimator object.
    bool enableCpuToGpuDescrAsyncCpy     = true;
    cuphySrsChEstAlgoType_t chEstAlgo    = SRS_CH_EST_ALGO_TYPE_MMSE;
    cuphySrsRkhsPrms_t*     pSrsRkhsPrms = nullptr;

    cuphyStatus_t status = cuphyCreateSrsChEst(&m_srsChEstHndl,
                                               &m_srsFilterPrms,
                                               pSrsRkhsPrms,
                                               chEstAlgo,
                                               enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                               m_statDescrBufCpu.addr(),
                                               m_statDescrBufGpu.addr(),
                                               m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateSrsChEst()");
    }
}


SrsChannelEstimator::~SrsChannelEstimator() {
    destroy();
}


const std::vector<cuphySrsChEstBuffInfo_t>& SrsChannelEstimator::estimate(
        const std::vector<cuphyTensorPrm_t>& tDataRx,
        const uint16_t nSrsUes,
        const uint16_t nCells,
        const uint16_t nPrbGrps,
        const uint16_t startPrbGrp,
        const std::vector<cuphySrsCellPrms_t>& srsCellPrms,
        const std::vector<cuphyUeSrsPrm_t>& ueSrsPrms) {

    m_nSrsUes = nSrsUes;
    m_nCells = nCells;

    m_linearAlloc.reset();
    m_tSrsChEstVec.clear();

    m_srsChEstBuffInfo.resize(nSrsUes);
    m_chEstCpuBuffVec.resize(nSrsUes);
    m_dChEstToL2Vec.resize(nSrsUes);
    m_chEstToL2Vec.resize(nSrsUes);
    m_srsRbSnrBuffOffsets.resize(nSrsUes);

    // Initializations.
    uint32_t rbSnrBufferSize = nSrsUes * m_nPrbs * sizeof(float);
    m_dSrsRbSnrBuffer = static_cast<float*>(m_linearAlloc.alloc(rbSnrBufferSize));
    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx) {

        uint16_t cellIdx = ueSrsPrms[ueIdx].cellIdx;
        uint16_t nRxAntSrs = srsCellPrms[cellIdx].nRxAntSrs;
        uint8_t prgSize = ueSrsPrms[ueIdx].prgSize;
        uint16_t nPrbGrpsPerHop = SRS_BW_TABLE[ueSrsPrms[ueIdx].configIdx][2 * ueSrsPrms[ueIdx].bandwidthIdx] / prgSize;
        uint16_t nHops = ueSrsPrms[ueIdx].nSyms / ueSrsPrms[ueIdx].nRepetitions;
        uint16_t nAntPorts = ueSrsPrms[ueIdx].nAntPorts;

        m_srsRbSnrBuffOffsets[ueIdx] = ueIdx * m_nPrbs;

        m_tSrsChEstVec.push_back(cuphy::tensor_device(CUPHY_C_32F,
                                                      nPrbGrps,
                                                      nRxAntSrs,
                                                      nAntPorts,
                                                      cuphy::tensor_flags::align_tight));

        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.desc = m_tSrsChEstVec[ueIdx].desc().handle();
        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.pAddr = m_tSrsChEstVec[ueIdx].addr();
        m_srsChEstBuffInfo[ueIdx].startPrbGrp = startPrbGrp;

        // Allocate buffers for ChEst to L2.
        size_t maxChEstSize = nPrbGrpsPerHop * nRxAntSrs * nHops * nAntPorts * sizeof(float2);
        m_chEstCpuBuffVec[ueIdx] = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(maxChEstSize));
        m_dChEstToL2Vec[ueIdx] = m_linearAlloc.alloc(maxChEstSize);
        m_chEstToL2Vec[ueIdx].pChEstCpuBuff = m_chEstCpuBuffVec[ueIdx].addr();

    }
    m_dSrsReports = static_cast<cuphySrsReport_t*>(m_linearAlloc.alloc(sizeof(cuphySrsReport_t) * nSrsUes));

    // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
    cuphySrsChEstLaunchCfg_t srsChEstLaunchCfg;

    // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
    bool  enableCpuToGpuDescrAsyncCpy = false;
    void* d_workspace                 = nullptr;
    cuphyStatus_t setupStatus = cuphySetupSrsChEst(m_srsChEstHndl,
                                                   nSrsUes,
                                                   const_cast<cuphyUeSrsPrm_t*>(ueSrsPrms.data()),
                                                   nCells,
                                                   const_cast<cuphyTensorPrm_t*>(tDataRx.data()),
                                                   const_cast<cuphySrsCellPrms_t*>(srsCellPrms.data()),
                                                   m_dSrsRbSnrBuffer,
                                                   m_srsRbSnrBuffOffsets.data(),
                                                   m_dSrsReports,
                                                   m_srsChEstBuffInfo.data(),
                                                   m_dChEstToL2Vec.data(),
                                                   m_chEstToL2Vec.data(),
                                                   d_workspace,
                                                   static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                   m_dynDescrBufCpu.addr(),
                                                   m_dynDescrBufGpu.addr(),
                                                   &srsChEstLaunchCfg,
                                                   m_cuStream);
    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(),
                                   m_dynDescrBufCpu.addr(),
                                   m_dynDescrSizeBytes,
                                   cudaMemcpyHostToDevice,
                                   m_cuStream));
    }

    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = srsChEstLaunchCfg.kernelNodeParamsDriver;
    CUresult srsChEstRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
                                                 kernelNodeParamsDriver.gridDimX,
                                                 kernelNodeParamsDriver.gridDimY,
                                                 kernelNodeParamsDriver.gridDimZ,
                                                 kernelNodeParamsDriver.blockDimX,
                                                 kernelNodeParamsDriver.blockDimY,
                                                 kernelNodeParamsDriver.blockDimZ,
                                                 kernelNodeParamsDriver.sharedMemBytes,
                                                 m_cuStream,
                                                 kernelNodeParamsDriver.kernelParams,
                                                 kernelNodeParamsDriver.extra);
    if(srsChEstRunStatus != CUDA_SUCCESS) {
        throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    }

    return m_srsChEstBuffInfo;
}


void SrsChannelEstimator::destroy() {
    // Destroy the SRS channel estimation handle.
    cuphyStatus_t status = cuphyDestroySrsChEst(m_srsChEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroySrsChEst()");
    }
}

} // namespace pycuphy
