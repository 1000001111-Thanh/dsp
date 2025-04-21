/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef PYCUPHY_CHAN_MODEL_HPP
#define PYCUPHY_CHAN_MODEL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cuda_runtime.h>
#include "fading_chan.cuh"  // include the fading channel header

namespace py = pybind11;

namespace pycuphy {
/*-------------------------------       OFDM modulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class OfdmModulateWrapper{
public:
    OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, py::array_t<std::complex<Tscalar>> freqDataInCpu, uintptr_t streamHandle); // constructor using Python array on CPU
    OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, uintptr_t freqDataInGpu, uintptr_t streamHandle);  // constructor using GPU memory pointer
    ~OfdmModulateWrapper();

    void run(py::array_t<std::complex<Tscalar>> freqDataInCpu = py::none()); // run function to enable changing input numpy array in run; if not provided, use the location in initialization
    void printTimeSample(int printLen = 10){ m_ofdmModulateHandle -> printTimeSample(printLen); }
    uintptr_t getTimeDataOut(){ return reinterpret_cast<uintptr_t>(m_ofdmModulateHandle -> getTimeDataOut()); }
    uint32_t getTimeDataLen(){ return m_ofdmModulateHandle -> getTimeDataLen(); }

private:
    ofdm_modulate::ofdmModulate<Tscalar, Tcomplex> * m_ofdmModulateHandle;
    cudaStream_t m_cuStrm;
    size_t m_freqDataInSize;
    Tcomplex* m_freqDataInCpu;
    Tcomplex* m_freqDataInGpu;
    uint8_t m_externGpuAlloc; // indicator for freqDataIn storage type: 0 - internal GPU memory allocation; 1 - external GPU memory allocation
};
// explicit instantiation
template class OfdmModulateWrapper<float, cuComplex>;

/*-------------------------------       OFDM demodulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class OfdmDeModulateWrapper{
public:
    OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, py::array_t<std::complex<Tscalar>> freqDataOutCpu, bool prach, uintptr_t streamHandle); // constructor using Python array on CPU, timeDataInGpu is always GPU memory address
    OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, uintptr_t freqDataOutGpu, bool prach, uintptr_t streamHandle); // constructor using GPU memory pointer, timeDataInGpu is always GPU memory address
    ~OfdmDeModulateWrapper();

    void run(py::array_t<std::complex<Tscalar>> freqDataOutCpu = py::none()); // run function to enable changing out numpy array in run, if not provided, use the location in initialization
    void printFreqSample(int printLen = 10){ m_ofdmDeModulateHandle -> printFreqSample(printLen); }
    uintptr_t getFreqDataOut(){return reinterpret_cast<uintptr_t>(m_ofdmDeModulateHandle -> getFreqDataOut()); } 

private:
    ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex> * m_ofdmDeModulateHandle;
    cudaStream_t m_cuStrm;
    size_t m_freqDataOutSize;
    Tcomplex *m_freqDataOutCpu;
    Tcomplex *m_freqDataOutGpu;
    uint8_t m_externGpuAlloc; // indicator for freqDataOut storage type: 0 - internal GPU memory allocation; 1 - external GPU memory allocation
};
// explicit instantiation
template class OfdmDeModulateWrapper<float, cuComplex>;

/*-------------------------------       TDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class TdlChanWrapper{
public:
    TdlChanWrapper(tdlConfig_t* tdlCfg, py::array_t<std::complex<Tscalar>> txTimeSigInCpu, uint16_t randSeed, uintptr_t streamHandle); // constructor using Python array on CPU
    TdlChanWrapper(tdlConfig_t* tdlCfg, uintptr_t txTimeSigInGpu, uint16_t randSeed, uintptr_t streamHandle); // constructor using GPU memory pointer
    ~TdlChanWrapper();

    // note: convert the channel pointer to uintptr_t for easy handling in Python
    void run(float refTime0 = 0.0f);
    uintptr_t getTdlTimeChan(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getTdlTimeChan()); }
    uintptr_t getTdlFreqChanSc(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getTdlFreqChanSc()); }
    uintptr_t getTdlFreqChanPrbg(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getTdlFreqChanPrbg()); }
    uintptr_t getRxTimeSigOut(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getRxTimeSigOut()); }
    uint32_t getTimeChanSize(){ return m_tdlChanHandle -> getTimeChanSize(); }
    uint32_t getFreqChanScPerLinkSize(){ return m_tdlChanHandle -> getFreqChanScPerLinkSize();}
    uint32_t getFreqChanPrbgSize(){ return m_tdlChanHandle -> getFreqChanPrbgSize(); }
    void printTimeChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printTimeChan(cid, uid, printLen); }
    void printFreqScChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printFreqScChan(cid, uid, printLen); }
    void printFreqPrbgChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printFreqPrbgChan(cid, uid, printLen); }
    void printTimeSig(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printTimeSig(cid, uid, printLen); }
    void printGpuMemUseMB(){ m_tdlChanHandle -> printGpuMemUseMB(); }

    /**
    * @brief This function saves the tdl data into h5 file, for verification in matlab using verify_tdl.m
    * 
    * @param padFileNameEnding optional ending of h5 file, e.g., tdlChan_1cell1Ue_4x4_A30_dopp10_cfo200_runMode0_FP32<padFileNameEnding>.h5
    */
    void saveTdlChanToH5File(std::string & padFileNameEnding = nullptr) {m_tdlChanHandle -> saveTdlChanToH5File(padFileNameEnding); };

private:
    tdlChan<Tscalar, Tcomplex> * m_tdlChanHandle;
    cudaStream_t m_cuStrm;
    size_t m_txSigSize;
    Tcomplex* m_txTimeSigInCpu;
    Tcomplex* m_txTimeSigInGpu;
    uint8_t m_externGpuAlloc; // indicator for txTimeSigIn storage type: 0 - internal GPU memory allocation; 1 - external GPU memory allocation
};
// explicit instantiation
template class TdlChanWrapper<float, cuComplex>;

} // namespace pycuphy

#endif // PYCUPHY_CHAN_MODEL_HPP