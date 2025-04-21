/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_SRS_CHEST_HPP
#define PYCUPHY_SRS_CHEST_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuphy.h"


namespace py = pybind11;

namespace pycuphy {


// This is the interface when called from C++.
class SrsChannelEstimator {
public:
    // Runs init() internally.
    SrsChannelEstimator(const cuphySrsFilterPrms_t& srsFilterPrms, cudaStream_t cuStream);

    // When using this, init() needs to be run separately, use this when the filter parameters
    // are not available upon instatiating the object.
    SrsChannelEstimator(cudaStream_t cuStream);

    ~SrsChannelEstimator();

    void init(const cuphySrsFilterPrms_t& srsFilterPrms);

    const std::vector<cuphySrsChEstBuffInfo_t>& estimate(const std::vector<cuphyTensorPrm_t>& tDataRx,
                                                         const uint16_t nSrsUes,
                                                         const uint16_t nCells,
                                                         const uint16_t nPrbGrps,
                                                         const uint16_t startPrbGrp,
                                                         const std::vector<cuphySrsCellPrms_t>& srsCellPrms,
                                                         const std::vector<cuphyUeSrsPrm_t>& ueSrsPrms);

    // Getters for other estimation results after running estimate().
    cuphySrsReport_t* getSrsReport() const { return m_dSrsReports; }
    float* getRbSnrBuffer() const { return m_dSrsRbSnrBuffer; }
    const std::vector<uint32_t>& getRbSnrBufferOffsets() const { return m_srsRbSnrBuffOffsets; }

private:
    size_t getBufferSize() const;
    void destroy();
    void allocateDescr();

    uint16_t m_nSrsUes;
    uint16_t m_nCells;
    uint16_t m_nPrbs;

    cuphy::linear_alloc<128, cuphy::device_alloc> m_linearAlloc;

    // Descriptor variables.
    size_t m_statDescrSizeBytes;
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_statDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_statDescrBufGpu;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    // SRS estimator handle.
    cuphySrsChEstHndl_t m_srsChEstHndl;

    cuphySrsFilterPrms_t m_srsFilterPrms;

    cudaStream_t m_cuStream;

    cuphySrsReport_t* m_dSrsReports;

    std::vector<cuphy::tensor_device> m_tSrsChEstVec;
    std::vector<cuphySrsChEstBuffInfo_t> m_srsChEstBuffInfo;

    std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>> m_chEstCpuBuffVec;
    std::vector<void*> m_dChEstToL2Vec;
    std::vector<cuphySrsChEstToL2_t> m_chEstToL2Vec;

    float* m_dSrsRbSnrBuffer;
    std::vector<uint32_t> m_srsRbSnrBuffOffsets;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PySrsChannelEstimator {
public:
    PySrsChannelEstimator(const py::dict& chEstParams, uint64_t cuStream);

    const std::vector<py::array_t<std::complex<float>>>& estimate(const py::array& inputData,
                                                                  const uint16_t nSrsUes,
                                                                  const uint16_t nCells,
                                                                  const uint16_t nPrbGrps,
                                                                  const uint16_t startPrbGrp,
                                                                  const std::vector<py::object>& srsCellPrms,
                                                                  const std::vector<py::object>& ueSrsPrms);

    const std::vector<cuphySrsReport_t>& getSrsReport() const { return m_srsReports; }
    py::array_t<float> getRbSnrBuffer() const;
    const std::vector<uint32_t>& getRbSnrBufferOffsets() { return m_srsChEstimator.getRbSnrBufferOffsets(); }

private:
    void readChEstParams(const py::dict& chEstParams);
    void readSrsCellParams(std::vector<cuphySrsCellPrms_t>& srsCellPrms, const std::vector<py::object>& pySrsCellPrms);
    void readUeSrsParams(std::vector<cuphyUeSrsPrm_t>& ueSrsPrms, const std::vector<py::object>& pyUeSrsPrms);

    SrsChannelEstimator m_srsChEstimator;

    uint16_t m_nSrsUes;
    uint16_t m_nPrbs;

    // Filter tensors and parameters.
    cuphy::tensor_device m_tPrmFocc_table;
    cuphy::tensor_device m_tPrmFocc_comb2_table;
    cuphy::tensor_device m_tPrmFocc_comb4_table;

    cuphy::tensor_device m_tPrmW_comb2_nPorts1_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts2_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts4_wide;
    cuphy::tensor_device m_tPrmW_comb2_nPorts8_wide;

    cuphy::tensor_device m_tPrmW_comb4_nPorts1_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts2_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts4_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts6_wide;
    cuphy::tensor_device m_tPrmW_comb4_nPorts12_wide;

    cuphy::tensor_device m_tPrmW_comb2_nPorts1_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts2_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts4_narrow;
    cuphy::tensor_device m_tPrmW_comb2_nPorts8_narrow;

    cuphy::tensor_device m_tPrmW_comb4_nPorts1_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts2_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts4_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts6_narrow;
    cuphy::tensor_device m_tPrmW_comb4_nPorts12_narrow;

    cuphySrsFilterPrms_t m_srsFilterPrms;

    cudaStream_t m_cuStream;

    // Outputs.
    std::vector<py::array_t<std::complex<float>>> m_chEsts;
    std::vector<cuphySrsReport_t> m_srsReports;
    std::vector<float> m_srsRbSnrBuffer;
};

} // namespace pycuphy

#endif // PYCUPHY_SRS_CHEST_HPP