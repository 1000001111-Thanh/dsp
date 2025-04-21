/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "pycuphy_chan_model.hpp"

namespace py = pybind11;

namespace pycuphy {

/*-------------------------------       OFDM modulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, py::array_t<std::complex<Tscalar>> freqDataInCpu, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_freqDataInSize = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_txLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    // Get buffer info from the NumPy array
    py::buffer_info buf = freqDataInCpu.request();
    m_freqDataInCpu = static_cast<Tcomplex*>(buf.ptr);
    assert(buf.size == m_freqDataInSize); // check data size match

    // allocate GPU buffer
    CUDA_CHECK(cudaMalloc((void**) &(m_freqDataInGpu), sizeof(Tcomplex) * m_freqDataInSize));
    m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, m_freqDataInGpu, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, uintptr_t freqDataInGpu, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(1)
{
    m_freqDataInGpu = (Tcomplex*)freqDataInGpu;
    m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, m_freqDataInGpu, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::~OfdmModulateWrapper()
{
    if(!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_freqDataInGpu);
    }
    delete m_ofdmModulateHandle;
}

template <typename Tscalar, typename Tcomplex>
void OfdmModulateWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> freqDataInCpu)
{
    if(freqDataInCpu.size() != 0) // new input numpy array, need to copy new data to GPU
    {
        // Get buffer info from the NumPy array
        py::buffer_info buf = freqDataInCpu.request();
        Tcomplex* freqDataInCpuNew = static_cast<Tcomplex*>(buf.ptr);
        assert(buf.size == m_freqDataInSize); // check data size match

        cudaMemcpyAsync(m_freqDataInGpu, freqDataInCpuNew, sizeof(Tcomplex) * m_freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm);
    }
    else
    {
        if(!m_externGpuAlloc) // use numpy array, need to copy new data to GPU
        {
            cudaMemcpyAsync(m_freqDataInGpu, m_freqDataInCpu, sizeof(Tcomplex) * m_freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm);
        }
    }
    m_ofdmModulateHandle -> run(m_cuStrm);
}

/*-------------------------------       OFDM demodulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, py::array_t<std::complex<Tscalar>> freqDataOutCpu, bool prach, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_freqDataOutSize = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_rxLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    // Get buffer info from the NumPy array
    py::buffer_info buf = freqDataOutCpu.request();
    m_freqDataOutCpu = static_cast<Tcomplex*>(buf.ptr);
    assert(buf.size == m_freqDataOutSize); // check data size match

    // allocate GPU buffer
    CUDA_CHECK(cudaMalloc((void**) &(m_freqDataOutGpu), sizeof(Tcomplex) * m_freqDataOutSize));
    m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, (Tcomplex*)timeDataInGpu, m_freqDataOutGpu, prach, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, uintptr_t freqDataOutGpu, bool prach, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(1)
{
    // buffer size from config
    m_freqDataOutSize = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_rxLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    m_freqDataOutCpu = nullptr;
    m_freqDataOutGpu = (Tcomplex*) freqDataOutGpu;
    m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, (Tcomplex*)timeDataInGpu, m_freqDataOutGpu, prach, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::~OfdmDeModulateWrapper()
{
    if(!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_freqDataOutGpu);
    }
    delete m_ofdmDeModulateHandle;
}

template <typename Tscalar, typename Tcomplex>
void OfdmDeModulateWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> freqDataOutCpu)
{
    m_ofdmDeModulateHandle -> run(m_cuStrm);
    if(freqDataOutCpu.size() != 0) // new output numpy array, need to copy new data from GPU
    {
        py::buffer_info buf = freqDataOutCpu.request();
        Tcomplex* freqDataOutCpuNew = static_cast<Tcomplex*>(buf.ptr);
        assert(buf.size == m_freqDataOutSize); // check data size match

        cudaMemcpyAsync(freqDataOutCpuNew, m_freqDataOutGpu, sizeof(Tcomplex) * m_freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
    else if(!m_externGpuAlloc)
    {
        cudaMemcpyAsync(m_freqDataOutCpu, m_freqDataOutGpu, sizeof(Tcomplex) * m_freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
    cudaStreamSynchronize(m_cuStrm);
}

/*-------------------------------       TDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::TdlChanWrapper(tdlConfig_t* tdlCfg, py::array_t<std::complex<Tscalar>> txTimeSigInCpu, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_txSigSize = (tdlCfg -> nCell) * (tdlCfg -> nUe) * (tdlCfg -> timeSigLenPerAnt) * (tdlCfg -> nTxAnt);
    // Get buffer info from the NumPy array
    py::buffer_info buf = txTimeSigInCpu.request();
    m_txTimeSigInCpu = static_cast<Tcomplex*>(buf.ptr);
    assert(buf.size == m_txSigSize); // check data size match

    // allocate GPU buffer and copy data
    CUDA_CHECK(cudaMalloc((void**) &(m_txTimeSigInGpu), sizeof(Tcomplex) * m_txSigSize));
    tdlCfg -> txTimeSigIn = m_txTimeSigInGpu;
    m_tdlChanHandle = new tdlChan<Tscalar, Tcomplex>(tdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::TdlChanWrapper(tdlConfig_t* tdlCfg, uintptr_t txTimeSigInGpu, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(1)
{
    m_txTimeSigInGpu = (cuComplex*) txTimeSigInGpu;
    tdlCfg -> txTimeSigIn = m_txTimeSigInGpu;
    m_tdlChanHandle = new tdlChan<Tscalar, Tcomplex>(tdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::~TdlChanWrapper()
{
    if(!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_txTimeSigInGpu);
    }
    delete m_tdlChanHandle;
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::run(float refTime0)
{
    if(!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaMemcpyAsync(m_txTimeSigInGpu, m_txTimeSigInCpu, sizeof(Tcomplex) * m_txSigSize, cudaMemcpyHostToDevice, m_cuStrm);
    }
    m_tdlChanHandle -> run(refTime0);
}

}