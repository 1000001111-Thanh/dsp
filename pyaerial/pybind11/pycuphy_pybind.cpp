/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "pycuphy_util.hpp"
#include "pycuphy_params.hpp"
#include "pycuphy_pusch.hpp"
#include "pycuphy_pdsch.hpp"
#include "pycuphy_csirs_tx.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_crc_check.hpp"
#include "pycuphy_channel_est.hpp"
#include "pycuphy_noise_intf_est.hpp"
#include "pycuphy_cfo_ta_est.hpp"
#include "pycuphy_channel_eq.hpp"
#include "pycuphy_srs_chest.hpp"
#include "pycuphy_trt_engine.hpp"
#include "pycuphy_rsrp.hpp"
#include "pycuphy_chan_model.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_pycuphy, m) {
    m.doc() = "Python bindings for cuPHY"; // optional module docstring

    m.def("device_to_numpy", &pycuphy::deviceToNumpy<std::complex<float>>);
    m.def("device_to_numpy", &pycuphy::deviceToNumpy<float>);

    // Enums here.
    py::enum_<cuphyPuschProcMode_t>(m, "PuschProcMode", py::arithmetic(), "PUSCH processing modes")
        .value("PUSCH_PROC_MODE_FULL_SLOT", cuphyPuschProcMode_t::PUSCH_PROC_MODE_FULL_SLOT)
        .value("PUSCH_PROC_MODE_FULL_SLOT_GRAPHS", cuphyPuschProcMode_t::PUSCH_PROC_MODE_FULL_SLOT_GRAPHS)
        .value("PUSCH_PROC_MODE_SUB_SLOT", cuphyPuschProcMode_t::PUSCH_PROC_MODE_SUB_SLOT)
        .value("PUSCH_MAX_PROC_MODES", cuphyPuschProcMode_t::PUSCH_MAX_PROC_MODES)
        .export_values();

    py::enum_<cuphyPuschLdpcKernelLaunch_t>(m, "PuschLdpcKernelLaunch", py::arithmetic(), "PUSCH kernel launch modes")
        .value("PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH)
        .value("PUSCH_RX_LDPC_STREAM_POOL", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_LDPC_STREAM_POOL)
        .value("PUSCH_RX_LDPC_STREAM_SEQUENTIAL", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
        .value("PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT", cuphyPuschLdpcKernelLaunch_t::PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT)
        .export_values();

    py::enum_<cuphyPuschWorkCancelMode_t>(m, "PuschWorkCancelMode", py::arithmetic(), "PUSCH work cancellation modes")
        .value("PUSCH_NO_WORK_CANCEL", cuphyPuschWorkCancelMode_t::PUSCH_NO_WORK_CANCEL)
        .value("PUSCH_COND_IF_NODES_W_KERNEL", cuphyPuschWorkCancelMode_t::PUSCH_COND_IF_NODES_W_KERNEL)
        .value("PUSCH_DEVICE_GRAPHS", cuphyPuschWorkCancelMode_t::PUSCH_DEVICE_GRAPHS)
        .value("PUSCH_MAX_WORK_CANCEL_MODES", cuphyPuschWorkCancelMode_t::PUSCH_MAX_WORK_CANCEL_MODES)
        .export_values();

    py::enum_<cuphyLdpcMaxItrAlgoType_t>(m, "LdpcMaxItrAlgoType", py::arithmetic(), "LDPC number of iterations algorithm types")
        .value("LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED", cuphyLdpcMaxItrAlgoType_t::LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED)
        .value("LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT", cuphyLdpcMaxItrAlgoType_t::LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT)
        .export_values();

    py::enum_<cuphyDataType_t>(m, "DataType", py::arithmetic(), "Data types")
        .value("CUPHY_VOID", cuphyDataType_t::CUPHY_VOID, "Uninitialized type")
        .value("CUPHY_BIT", cuphyDataType_t::CUPHY_BIT, "1-bit value")
        .value("CUPHY_R_8I", cuphyDataType_t::CUPHY_R_8I, "8-bit signed integer real values")
        .value("CUPHY_C_8I", cuphyDataType_t::CUPHY_C_8I, "8-bit signed integer complex values")
        .value("CUPHY_R_8U", cuphyDataType_t::CUPHY_R_8U, "8-bit unsigned integer real values")
        .value("CUPHY_C_8U", cuphyDataType_t::CUPHY_C_8U, "8-bit unsigned integer complex values")
        .value("CUPHY_R_16I", cuphyDataType_t::CUPHY_R_16I, "16-bit signed integer real values")
        .value("CUPHY_C_16I", cuphyDataType_t::CUPHY_C_16I, "16-bit signed integer complex values")
        .value("CUPHY_R_16U", cuphyDataType_t::CUPHY_R_16U, "16-bit unsigned integer real values")
        .value("CUPHY_C_16U", cuphyDataType_t::CUPHY_C_16U, "16-bit unsigned integer complex values")
        .value("CUPHY_R_32I", cuphyDataType_t::CUPHY_R_32I, "32-bit signed integer real values")
        .value("CUPHY_C_32I", cuphyDataType_t::CUPHY_C_32I, "32-bit signed integer complex values")
        .value("CUPHY_R_32U", cuphyDataType_t::CUPHY_R_32U, "32-bit unsigned integer real values")
        .value("CUPHY_C_32U", cuphyDataType_t::CUPHY_C_32U, "32-bit unsigned integer complex values")
        .value("CUPHY_R_16F", cuphyDataType_t::CUPHY_R_16F, "Half precision (16-bit) real values")
        .value("CUPHY_C_16F", cuphyDataType_t::CUPHY_C_16F, "Half precision (16-bit) complex values")
        .value("CUPHY_R_32F", cuphyDataType_t::CUPHY_R_32F, "Single precision (32-bit) real values")
        .value("CUPHY_C_32F", cuphyDataType_t::CUPHY_C_32F, "Single precision (32-bit) complex values")
        .value("CUPHY_R_64F", cuphyDataType_t::CUPHY_R_64F, "Double precision (64-bit) real values")
        .value("CUPHY_C_64F", cuphyDataType_t::CUPHY_C_64F, "Double precision (64-bit) complex values")
        .export_values();

    py::enum_<cuphyPuschSetupPhase_t>(m, "PuschSetupPhase", py::arithmetic(), "PUSCH setup phases")
        .value("PUSCH_SETUP_PHASE_INVALID", cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_INVALID)
        .value("PUSCH_SETUP_PHASE_1", cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_1)
        .value("PUSCH_SETUP_PHASE_2", cuphyPuschSetupPhase_t::PUSCH_SETUP_PHASE_2)
        .value("PUSCH_SETUP_MAX_PHASES", cuphyPuschSetupPhase_t::PUSCH_SETUP_MAX_PHASES)
        .value("PUSCH_SETUP_MAX_VALID_PHASES", cuphyPuschSetupPhase_t::PUSCH_SETUP_MAX_VALID_PHASES)
        .export_values();

    py::enum_<cuphyPuschRunPhase_t>(m, "PuschRunPhase", py::arithmetic(), "PUSCH run phases")
        .value("PUSCH_RUN_PHASE_INVALID", cuphyPuschRunPhase_t::PUSCH_RUN_PHASE_INVALID)
        .value("PUSCH_RUN_SUB_SLOT_PROC", cuphyPuschRunPhase_t::PUSCH_RUN_SUB_SLOT_PROC)
        .value("PUSCH_RUN_FULL_SLOT_COPY", cuphyPuschRunPhase_t::PUSCH_RUN_FULL_SLOT_COPY)
        .value("PUSCH_RUN_ALL_PHASES", cuphyPuschRunPhase_t::PUSCH_RUN_ALL_PHASES)
        .value("PUSCH_RUN_MAX_PHASES", cuphyPuschRunPhase_t::PUSCH_RUN_MAX_PHASES)
        .value("PUSCH_RUN_MAX_VALID_PHASES", cuphyPuschRunPhase_t::PUSCH_RUN_MAX_VALID_PHASES)
        .export_values();

    py::enum_<cuphyPuschEqCoefAlgoType_t>(m, "PuschEqCoefAlgoType", py::arithmetic(), "PUSCH equalizer algorithm types")
        .value("PUSCH_EQ_ALGO_TYPE_RZF", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_RZF)
        .value("PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)
        .value("PUSCH_EQ_ALGO_TYPE_MMSE_IRC", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_MMSE_IRC)
        .value("PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW)
        .value("PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS)
        .value("PUSCH_EQ_ALGO_MAX_TYPES", cuphyPuschEqCoefAlgoType_t::PUSCH_EQ_ALGO_MAX_TYPES)
        .export_values();

    py::enum_<cuphyPuschStatusType_t>(m, "PuschStatusType", py::arithmetic(), "PUSCH status types")
        .value("CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE", cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE)
        .value("CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB", cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB)
        .value("CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH", cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH)
        .value("CUPHY_MAX_PUSCH_STATUS_TYPES", cuphyPuschStatusType_t::CUPHY_MAX_PUSCH_STATUS_TYPES)
        .export_values();

    py::enum_<cuphyPuschChEstAlgoType_t>(m, "PuschChEstAlgoType", py::arithmetic(), "PUSCH channel estimation algorithm types")
        .value("PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE", cuphyPuschChEstAlgoType_t::PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE)
        .value("PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST", cuphyPuschChEstAlgoType_t::PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST)
        .value("PUSCH_CH_EST_ALGO_TYPE_RKHS", cuphyPuschChEstAlgoType_t::PUSCH_CH_EST_ALGO_TYPE_RKHS)
        .export_values();

    // Full channel pipelines.
    py::class_<pycuphy::PdschPipeline>(m, "PdschPipeline")
        .def(py::init<const py::object&>())
        .def("setup_pdsch_tx", &pycuphy::PdschPipeline::setupPdschTx)
        .def("run_pdsch_tx", &pycuphy::PdschPipeline::runPdschTx)
        .def("get_ldpc_output", &pycuphy::PdschPipeline::getLdpcOutputPerTbPerCell);

    py::class_<pycuphy::PuschPipeline>(m, "PuschPipeline")
        .def(py::init<const py::object&, uint64_t>())
        .def("setup_pusch_rx", &pycuphy::PuschPipeline::setupPuschRx)
        .def("run_pusch_rx", &pycuphy::PuschPipeline::runPuschRx)
        .def("write_dbg_buf_synch", &pycuphy::PuschPipeline::writeDbgBufSynch);

    // Individual Tx/Rx components.
    py::class_<pycuphy::PyCsiRsTx>(m, "CsiRsTx")
        .def(py::init<const std::vector<uint16_t>&>())
        .def("run", &pycuphy::PyCsiRsTx::run);

    py::class_<pycuphy::LdpcEncoder>(m, "LdpcEncoder")
        .def(py::init<const uint64_t, const uint64_t, const uint64_t, const uint64_t, const uint64_t>())
        .def("encode", &pycuphy::LdpcEncoder::encode)
        .def("set_profiling_iterations", &pycuphy::LdpcEncoder::setProfilingIterations)
        .def("set_puncturing", &pycuphy::LdpcEncoder::setPuncturing);

    py::class_<pycuphy::PyLdpcDecoder>(m, "LdpcDecoder")
        .def(py::init<const uint64_t>())
        .def("decode", &pycuphy::PyLdpcDecoder::decode)
        .def("set_num_iterations", &pycuphy::PyLdpcDecoder::setNumIterations)
        .def("get_soft_outputs", &pycuphy::PyLdpcDecoder::getSoftOutputs)
        .def("set_throughput_mode", &pycuphy::PyLdpcDecoder::setThroughputMode);

    py::class_<pycuphy::LdpcRateMatch>(m, "LdpcRateMatch")
        .def(py::init<const uint64_t, const uint64_t, const uint64_t, const uint64_t, const uint64_t, const bool, const uint64_t>())
        .def("rate_match", &pycuphy::LdpcRateMatch::rateMatch)
        .def("set_profiling_iterations", &pycuphy::LdpcRateMatch::setProfilingIterations);

    py::class_<pycuphy::PyLdpcDerateMatch>(m, "LdpcDerateMatch")
        .def(py::init<const bool, const uint64_t>())
        .def("derate_match", &pycuphy::PyLdpcDerateMatch::derateMatch);

    py::class_<pycuphy::PyCrcChecker>(m, "CrcChecker")
        .def(py::init<const uint64_t>())
        .def("check_crc", &pycuphy::PyCrcChecker::checkCrc)
        .def("get_tb_crcs", &pycuphy::PyCrcChecker::getTbCrcs)
        .def("get_cb_crcs", &pycuphy::PyCrcChecker::getCbCrcs);

    py::class_<pycuphy::PySrsChannelEstimator>(m, "SrsChannelEstimator")
        .def(py::init<const py::dict&, uint64_t>())
        .def("estimate", &pycuphy::PySrsChannelEstimator::estimate)
        .def("get_srs_report", &pycuphy::PySrsChannelEstimator::getSrsReport)
        .def("get_rb_snr_buffer", &pycuphy::PySrsChannelEstimator::getRbSnrBuffer)
        .def("get_rb_snr_buffer_offsets", &pycuphy::PySrsChannelEstimator::getRbSnrBufferOffsets);

    py::class_<cuphySrsReport_t>(m, "SrsReport")  // A read-only struct for passing the SRS reports.
        .def(py::init<>())
        .def_property_readonly("to_est_ms", [](const cuphySrsReport_t& prm) { return prm.toEstMicroSec; })
        .def_property_readonly("wideband_snr", [](const cuphySrsReport_t& prm) { return prm.widebandSnr; })
        .def_property_readonly("wideband_noise_energy", [](const cuphySrsReport_t& prm) { return prm.widebandNoiseEnergy; })
        .def_property_readonly("wideband_signal_energy", [](const cuphySrsReport_t& prm) { return prm.widebandSignalEnergy; })
        .def_property_readonly("wideband_sc_corr", [](const cuphySrsReport_t& prm) { return std::complex<float>(__high2float(prm.widebandScCorr), __low2float(prm.widebandScCorr)); })
        .def_property_readonly("wideband_cs_corr_ratio_db", [](const cuphySrsReport_t& prm) { return prm.widebandCsCorrRatioDb; })
        .def_property_readonly("wideband_cs_corr_use", [](const cuphySrsReport_t& prm) { return prm.widebandCsCorrUse; })
        .def_property_readonly("wideband_cs_corr_not_use", [](const cuphySrsReport_t& prm) { return prm.widebandCsCorrNotUse; });

    py::class_<pycuphy::PyChannelEstimator>(m, "ChannelEstimator")
        .def(py::init<const pycuphy::PuschParams&, const uint64_t>())
        .def("estimate", py::overload_cast<pycuphy::PuschParams&>(&pycuphy::PyChannelEstimator::estimate));

    py::class_<pycuphy::PyNoiseIntfEstimator>(m, "NoiseIntfEstimator")
        .def(py::init<const uint64_t>())
        .def("estimate", py::overload_cast<const std::vector<py::array_t<std::complex<float>>>&, pycuphy::PuschParams&>(&pycuphy::PyNoiseIntfEstimator::estimate))
        .def("get_info_noise_var_pre_eq", &pycuphy::PyNoiseIntfEstimator::getInfoNoiseVarPreEq)
        .def("get_inv_noise_var_lin", &pycuphy::PyNoiseIntfEstimator::getInvNoiseVarLin);

    py::class_<pycuphy::PyChannelEqualizer>(m, "ChannelEqualizer")
        .def(py::init<const uint64_t>())
        .def("equalize", py::overload_cast<const std::vector<py::array_t<std::complex<float>>>&, const std::vector<py::array_t<std::complex<float>>>&,
                                           const py::array&, const py::array&, pycuphy::PuschParams&>(&pycuphy::PyChannelEqualizer::equalize))
        .def("get_data_eq", &pycuphy::PyChannelEqualizer::getDataEq)
        .def("get_eq_coef", &pycuphy::PyChannelEqualizer::getEqCoef)
        .def("get_ree_diag_inv", &pycuphy::PyChannelEqualizer::getReeDiagInv);

    py::class_<pycuphy::PyCfoTaEstimator>(m, "CfoTaEstimator")
        .def(py::init<const uint64_t>())
        .def("estimate", py::overload_cast<const std::vector<py::array_t<std::complex<float>>>&, pycuphy::PuschParams&>(&pycuphy::PyCfoTaEstimator::estimate))
        .def("get_cfo_hz", &pycuphy::PyCfoTaEstimator::getCfoHz)
        .def("get_ta", &pycuphy::PyCfoTaEstimator::getTaEst)
        .def("get_cfo_phase_rot", &pycuphy::PyCfoTaEstimator::getCfoPhaseRot)
        .def("get_ta_phase_rot", &pycuphy::PyCfoTaEstimator::getTaPhaseRot);
    py::class_<pycuphy::PyRsrpEstimator>(m, "RsrpEstimator")
        .def(py::init<const uint64_t>())
        .def("estimate", py::overload_cast<const std::vector<py::array_t<std::complex<float>>>&, const std::vector<py::array_t<float>>&,
                                           const py::array&, pycuphy::PuschParams&>(&pycuphy::PyRsrpEstimator::estimate))
        .def("get_info_noise_var_post_eq", &pycuphy::PyRsrpEstimator::getInfoNoiseVarPostEq)
        .def("get_sinr_pre_eq", &pycuphy::PyRsrpEstimator::getSinrPreEq)
        .def("get_sinr_post_eq", &pycuphy::PyRsrpEstimator::getSinrPostEq);

    py::class_<pycuphy::PuschParams>(m, "PuschParams")
        .def(py::init<>())
        .def("set_filters", &pycuphy::PuschParams::setFilters)
        .def("print_stat_prms", &pycuphy::PuschParams::printStatPrms)
        .def("print_dyn_prms", &pycuphy::PuschParams::printDynPrms)
        .def("set_dyn_prms", py::overload_cast<const py::object&>(&pycuphy::PuschParams::setDynPrms))
        .def("set_stat_prms", py::overload_cast<const py::object&>(&pycuphy::PuschParams::setStatPrms));

    py::class_<pycuphy::PdschParams>(m, "PdschParams")
        .def(py::init<const py::object&>())
        .def("print_stat_prms", &pycuphy::PdschParams::printStatPrms)
        .def("set_dyn_prms", &pycuphy::PdschParams::setDynPrms);

    py::class_<pycuphy::PyTrtEngine>(m, "TrtEngine")
        .def(py::init<const std::string&,
                      const uint32_t,
                      const std::vector<std::string>&,
                      const std::vector<std::vector<int>>&,
                      const std::vector<cuphyDataType_t>&,
                      const std::vector<std::string>&,
                      const std::vector<std::vector<int>>&,
                      const std::vector<cuphyDataType_t>&,
                      uint64_t>())
        .def("run", py::overload_cast<const std::vector<py::array>&>(&pycuphy::PyTrtEngine::run));

    // carrier configuration
    py::class_<cuphyCarrierPrms_t>(m, "CuphyCarrierPrms")
        .def(py::init<>())
        .def_readwrite("n_sc", &cuphyCarrierPrms_t::N_sc)
        .def_readwrite("n_fft", &cuphyCarrierPrms_t::N_FFT)
        .def_readwrite("n_tx_layer", &cuphyCarrierPrms_t::N_txLayer)
        .def_readwrite("n_rx_layer", &cuphyCarrierPrms_t::N_rxLayer)
        .def_readwrite("id_slot", &cuphyCarrierPrms_t::id_slot)
        .def_readwrite("id_subframe", &cuphyCarrierPrms_t::id_subFrame)
        .def_readwrite("mu", &cuphyCarrierPrms_t::mu)
        .def_readwrite("cp_type", &cuphyCarrierPrms_t::cpType)
        .def_readwrite("f_c", &cuphyCarrierPrms_t::f_c)
        .def_readwrite("t_c", &cuphyCarrierPrms_t::T_c)
        .def_readwrite("f_samp", &cuphyCarrierPrms_t::f_samp)
        .def_readwrite("n_symbol_slot", &cuphyCarrierPrms_t::N_symbol_slot)
        .def_readwrite("k_const", &cuphyCarrierPrms_t::k_const)
        .def_readwrite("kappa_bits", &cuphyCarrierPrms_t::kappa_bits)
        .def_readwrite("ofdm_window_len", &cuphyCarrierPrms_t::ofdmWindowLen)
        .def_readwrite("rolloff_factor", &cuphyCarrierPrms_t::rolloffFactor)
        .def_readwrite("n_samp_slot", &cuphyCarrierPrms_t::N_samp_slot)

        // below are PRACH parameters
        .def_readwrite("n_u_mu", &cuphyCarrierPrms_t::N_u_mu)
        .def_readwrite("start_ra_sym", &cuphyCarrierPrms_t::startRaSym)
        .def_readwrite("delta_f_ra", &cuphyCarrierPrms_t::delta_f_RA)
        .def_readwrite("n_cp_ra", &cuphyCarrierPrms_t::N_CP_RA)
        .def_readwrite("k", &cuphyCarrierPrms_t::K)
        .def_readwrite("k1", &cuphyCarrierPrms_t::k1)
        .def_readwrite("k_bar", &cuphyCarrierPrms_t::kBar)
        .def_readwrite("n_u", &cuphyCarrierPrms_t::N_u)
        .def_readwrite("l_ra", &cuphyCarrierPrms_t::L_RA)
        .def_readwrite("n_slot_ra_sel", &cuphyCarrierPrms_t::n_slot_RA_sel)
        .def_readwrite("n_rep", &cuphyCarrierPrms_t::N_rep);

    // OFDM modulation
    py::class_<pycuphy::OfdmModulateWrapper<float, cuComplex>>(m, "OfdmModulate")
        .def(py::init<cuphyCarrierPrms_t*, uintptr_t, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("freq_data_in_gpu"), py::arg("stream_handle"))
        .def(py::init<cuphyCarrierPrms_t*, py::array_t<std::complex<float>>, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("freq_data_in_cpu"), py::arg("stream_handle"))
        .def("run", &pycuphy::OfdmModulateWrapper<float, cuComplex>::run,
            py::arg("freq_data_in_cpu") = py::array_t<std::complex<float>>())
        .def("print_time_sample", &pycuphy::OfdmModulateWrapper<float, cuComplex>::printTimeSample, py::arg("print_length") = 10)
        .def("get_time_data_out", &pycuphy::OfdmModulateWrapper<float, cuComplex>::getTimeDataOut, py::return_value_policy::reference)
        .def("get_time_data_length", &pycuphy::OfdmModulateWrapper<float, cuComplex>::getTimeDataLen);

    // OFDM demodulation
    py::class_<pycuphy::OfdmDeModulateWrapper<float, cuComplex>>(m, "OfdmDeModulate")
        .def(py::init<cuphyCarrierPrms_t*, uintptr_t, uintptr_t, bool, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("time_data_in_gpu"), py::arg("freq_data_out_gpu"), py::arg("prach"), py::arg("stream_handle"))
        .def(py::init<cuphyCarrierPrms_t*, uintptr_t, py::array_t<std::complex<float>>, bool, uintptr_t>(),
            py::arg("cuphy_carrier_prms"), py::arg("time_data_in_gpu"), py::arg("freq_data_out_cpu"), py::arg("prach"), py::arg("stream_handle"))
        .def("run", &pycuphy::OfdmDeModulateWrapper<float, cuComplex>::run,
            py::arg("freq_data_out_cpu") = py::array_t<std::complex<float>>())
        .def("print_freq_sample", &pycuphy::OfdmDeModulateWrapper<float, cuComplex>::printFreqSample, py::arg("print_length") = 10)
        .def("get_freq_data_out", &pycuphy::OfdmDeModulateWrapper<float, cuComplex>::getFreqDataOut, py::return_value_policy::reference);

    // TDL channel configuration
    py::class_<tdlConfig_t>(m, "TdlConfig")
        .def(py::init<>())
        .def_readwrite("use_simplified_pdp", &tdlConfig_t::useSimplifiedPdp)
        .def_readwrite("delay_profile", &tdlConfig_t::delayProfile)
        .def_readwrite("delay_spread", &tdlConfig_t::delaySpread)
        .def_readwrite("max_doppler_shift", &tdlConfig_t::maxDopplerShift)
        .def_readwrite("f_samp", &tdlConfig_t::f_samp)
        .def_readwrite("n_cell", &tdlConfig_t::nCell)
        .def_readwrite("n_ue", &tdlConfig_t::nUe)
        .def_readwrite("n_tx_ant", &tdlConfig_t::nTxAnt)
        .def_readwrite("n_rx_ant", &tdlConfig_t::nRxAnt)
        .def_readwrite("f_batch", &tdlConfig_t::fBatch)
        .def_readwrite("n_path", &tdlConfig_t::numPath)
        .def_readwrite("cfo_hz", &tdlConfig_t::cfoHz)
        .def_readwrite("delay", &tdlConfig_t::delay)
        .def_readwrite("time_signal_length_per_ant", &tdlConfig_t::timeSigLenPerAnt)
        .def_readwrite("n_sc", &tdlConfig_t::N_sc)
        .def_readwrite("n_sc_prbg", &tdlConfig_t::N_sc_Prbg)
        .def_readwrite("sc_spacing_hz", &tdlConfig_t::scSpacingHz)
        .def_readwrite("carrier_freq_hz", &tdlConfig_t::carrierFreqHz)
        .def_readwrite("freq_convert_type", &tdlConfig_t::freqConvertType)
        .def_readwrite("sc_sampling", &tdlConfig_t::scSampling)
        .def_readwrite("run_mode", &tdlConfig_t::runMode)
        .def_readwrite("tx_time_signal_in", &tdlConfig_t::txTimeSigIn);

    py::class_<pycuphy::TdlChanWrapper<float, cuComplex>>(m, "TdlChan")
        .def(py::init<tdlConfig_t*, uintptr_t, uint16_t, uintptr_t>(),
            py::arg("tdl_cfg"), py::arg("tx_time_signal_in_gpu"), py::arg("rand_seed"), py::arg("stream_handle"))
        .def(py::init<tdlConfig_t*, py::array_t<std::complex<float>>, uint16_t, uintptr_t>(),
            py::arg("tdl_cfg"), py::arg("tx_time_signal_in_cpu"), py::arg("rand_seed"), py::arg("stream_handle"))
        .def("run", &pycuphy::TdlChanWrapper<float, cuComplex>::run)
        .def("get_tdl_time_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::getTdlTimeChan)
        .def("get_tdl_freq_chan_sc", &pycuphy::TdlChanWrapper<float, cuComplex>::getTdlFreqChanSc)
        .def("get_tdl_freq_chan_prbg", &pycuphy::TdlChanWrapper<float, cuComplex>::getTdlFreqChanPrbg)
        .def("get_rx_time_signal_out", &pycuphy::TdlChanWrapper<float, cuComplex>::getRxTimeSigOut)
        .def("get_time_chan_size", &pycuphy::TdlChanWrapper<float, cuComplex>::getTimeChanSize)
        .def("get_freq_chan_sc_per_link_size", &pycuphy::TdlChanWrapper<float, cuComplex>::getFreqChanScPerLinkSize)
        .def("get_freq_chan_prbg_size", &pycuphy::TdlChanWrapper<float, cuComplex>::getFreqChanPrbgSize)
        .def("print_time_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::printTimeChan,
             py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_freq_sc_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::printFreqScChan,
             py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_freq_prbg_chan", &pycuphy::TdlChanWrapper<float, cuComplex>::printFreqPrbgChan,
             py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_time_signal", &pycuphy::TdlChanWrapper<float, cuComplex>::printTimeSig,
             py::arg("cid") = 0, py::arg("uid") = 0, py::arg("print_length") = 10)
        .def("print_gpu_memory_usage_mb", &pycuphy::TdlChanWrapper<float, cuComplex>::printGpuMemUseMB)
        .def("save_tdl_chan_to_h5_file", &pycuphy::TdlChanWrapper<float, cuComplex>::saveTdlChanToH5File,
            py::arg("pad_file_name_ending") = "");
 }
