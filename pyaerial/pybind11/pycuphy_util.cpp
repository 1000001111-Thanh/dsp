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

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "pycuphy_util.hpp"
#include "tensor_desc.hpp"

namespace py = pybind11;

namespace pycuphy {


template <typename T>
py::array_t<T> deviceToNumpy(uint64_t deviceAddr,
                             uint64_t hostAddr,
                             py::list dimensions,
                             uint64_t cuStream) {
    int nDim = dimensions.size();

    // T needs to be either std::complex<float> or float.
    // TODO: Fix this hack where we need to determine types based on T as it restricts
    // the use of this function.
    cuphyDataType_t deviceDataType, hostDataType;
    if(std::is_same<T, std::complex<float>>::value) {
        deviceDataType = CUPHY_C_16F;
        hostDataType = CUPHY_C_32F;
    }
    else if(std::is_same<T, float>::value) {
        deviceDataType = CUPHY_R_32F;
        hostDataType = CUPHY_R_32F;
    }
    else {
        throw std::runtime_error("deviceToNumpy: Unsupported data type!");
    }

    if(nDim == 2) {
        int dim0 = dimensions[0].cast<int>();
        int dim1 = dimensions[1].cast<int>();

        cuphy::tensor_device deviceTensor = cuphy::tensor_device((void*)deviceAddr, deviceDataType, dim0, dim1, cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hostTensor = cuphy::tensor_pinned((void*)hostAddr, hostDataType, dim0, dim1, cuphy::tensor_flags::align_tight);

        hostTensor.convert(deviceTensor, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
        return hostToNumpy<T>((T*)hostAddr, dim0, dim1);
    }

    else if(nDim ==3) {
        int dim0 = dimensions[0].cast<int>();
        int dim1 = dimensions[1].cast<int>();
        int dim2 = dimensions[2].cast<int>();

        cuphy::tensor_device deviceTensor = cuphy::tensor_device((void*)deviceAddr, deviceDataType, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);
        cuphy::tensor_pinned hostTensor = cuphy::tensor_pinned((void*)hostAddr, hostDataType, dim0, dim1, dim2, cuphy::tensor_flags::align_tight);

        hostTensor.convert(deviceTensor, (cudaStream_t)cuStream);
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)cuStream));
        return hostToNumpy<T>((T*)hostAddr, dim0, dim1, dim2);
    }

    else {
        throw std::runtime_error("\nInvalid tensor dimensions!\n");
    }
}

// Explicit instantiations.
template py::array_t<float> deviceToNumpy(uint64_t deviceAddr,
                                          uint64_t hostAddr,
                                          py::list dimensions,
                                          uint64_t cuStream);
template py::array_t<std::complex<float>> deviceToNumpy(uint64_t deviceAddr,
                                                        uint64_t hostAddr,
                                                        py::list dimensions,
                                                        uint64_t cuStream);


template <typename T>
void fromNumpyBitArray(float* src, T* dst, uint32_t npDim0, uint32_t npDim1) {

    static_assert(std::is_integral<T>::value, "Integral destination required.");

    const int ELEMENT_SIZE = sizeof(T) * 8;

    for(uint32_t col = 0; col < npDim1; col++) {
        for(int row = 0; row < npDim0; row += ELEMENT_SIZE) {
            T bits = 0;

            for(int o = 0; o < ELEMENT_SIZE; o++) {
                if(row + o < npDim0) {
                    float bit = *(src + (npDim1 * (row + o) + col));
                    T bit_0 = (T)bit & 0x1;
                    bits |= (bit_0 << o);
                }
            }

            // Target address. Set the data.
            T* dstElem = dst + (row / ELEMENT_SIZE) + (npDim0 / ELEMENT_SIZE) * col;
            *dstElem = bits;
        }
    }
}

// Explicit instantiations.
template void fromNumpyBitArray(float* src, uint32_t* dst, uint32_t npDim0, uint32_t npDim1);


template <typename T>
void toNumpyBitArray(T* src, float* dst, uint32_t dstDim0, uint32_t dstDim1) {

    static_assert(std::is_integral<T>::value, "Integral source required.");

    const int ELEMENT_SIZE = sizeof(T) * 8;

    uint32_t srcDim0 = (dstDim0 + ELEMENT_SIZE - 1) / ELEMENT_SIZE;
    for(uint32_t col = 0; col < dstDim1; col++) {
        for(uint32_t row = 0; row < dstDim0; row += ELEMENT_SIZE) {
            T* srcElem = src + srcDim0 * col + (row / ELEMENT_SIZE);
            for(int o = 0; o < ELEMENT_SIZE && (row + o < dstDim0); o++) {
                T bit = ((*srcElem & (1 << o)) >> o) & 1;
                float* dstElem = dst + dstDim1 * (row + o) + col;
                *dstElem = (float)bit;
            }
        }
    }
}

// Explicit instantiations.
template void toNumpyBitArray(uint32_t* src, float* dst, uint32_t dstDim0, uint32_t dstDim1);
template void toNumpyBitArray(uint8_t* src, float* dst, uint32_t dstDim0, uint32_t dstDim1);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0) {
    return py::array_t<T>(
        {dim0},  // Shape
        {sizeof(T)},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1) {
    return py::array_t<T>(
        {dim0, dim1},  // Shape
        {sizeof(T), sizeof(T) * dim0},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2) {
    return py::array_t<T>(
        {dim0, dim1, dim2},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3) {
    return py::array_t<T>(
        {dim0, dim1, dim2, dim3},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1, sizeof(T) * dim0 * dim1 * dim2},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3);


template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4) {
    return py::array_t<T>(
        {dim0, dim1, dim2, dim3, dim4},  // Shape
        {sizeof(T), sizeof(T) * dim0, sizeof(T) * dim0 * dim1, sizeof(T) * dim0 * dim1 * dim2, sizeof(T) * dim0 * dim1 * dim2 * dim3},  // Strides (in bytes) for each index
        dataPtr  // The data pointer
    );
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4);

template <typename T>
py::array_t<T> hostToNumpy(T* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides) {
    std::vector<size_t> strides_;
    // Default strides.
    if(strides.empty()) {
        strides_.push_back(sizeof(T));
        for(int i = 1; i < dims.size(); i++) {
            size_t stride = strides_[i - 1] * dims[i - 1];
            strides_.push_back(stride);
        }
    }
    else {
        strides_ = strides;
    }
    return py::array_t<T>(dims, strides_, dataPtr);
}

// Explicit instantiations.
template py::array_t<float> hostToNumpy(float* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);
template py::array_t<int> hostToNumpy(int* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);
template py::array_t<std::complex<float>> hostToNumpy(std::complex<float>* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);

template <typename T, int flags>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags hostTensorFlags,
                                     cuphy::tensor_flags deviceTensorFlags,
                                     cudaStream_t cuStream) {
    py::array_t<T, flags> array = py_array;
    py::buffer_info buf = array.request();

    int dims[buf.ndim];
    std::transform(buf.shape.begin(), buf.shape.end(), dims, [](const ssize_t & shape) { return static_cast<int>(shape); });
    int strides[buf.ndim];
    std::transform(buf.strides.begin(), buf.strides.end(), strides, [](const ssize_t & stride) { return static_cast<int>(stride); });

    cuphy::tensor_layout layout(buf.ndim, dims, strides);
    cuphy::tensor_info info(convertToType, layout);
    cuphy::tensor_pinned hostTensor(convertFromType, layout, hostTensorFlags);
    cuphy::tensor_device deviceTensor(inputDevPtr, info, deviceTensorFlags);

    // Copy the array to pinned memory first.
    size_t nBytes = buf.size * sizeof(T);
    memcpy(hostTensor.addr(), buf.ptr, nBytes);

    // Obtain a tensor in device memory.
    deviceTensor.convert(hostTensor, cuStream);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    return deviceTensor;
}

// Explicit instantiations.
template cuphy::tensor_device deviceFromNumpy<float>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<uint8_t>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags hostTensorFlags,
    cuphy::tensor_flags deviceTensorFlags,
    cudaStream_t cuStream);


template <typename T, int flags>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream) {
    return deviceFromNumpy<T, flags>(py_array,
                                     inputDevPtr,
                                     convertFromType,
                                     convertToType,
                                     cuphy::tensor_flags::align_tight,
                                     tensorDescFlags,
                                     cuStream);
}


// Explicit instantiations.
template cuphy::tensor_device deviceFromNumpy<float>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<uint8_t>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<__half, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<int, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    void * inputDevPtr,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);


template <typename T, int flags>
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream) {
    py::array_t<T, flags> array = py_array;
    py::buffer_info buf = array.request();

    int dims[buf.ndim];
    std::transform(buf.shape.begin(), buf.shape.end(), dims, [](const ssize_t & shape) { return static_cast<int>(shape); });
    int strides[buf.ndim];
    std::transform(buf.strides.begin(), buf.strides.end(), strides, [](const ssize_t & stride) { return static_cast<int>(stride); });

    cuphy::tensor_layout layout(buf.ndim, dims, strides);
    cuphy::tensor_pinned hostTensor(convertFromType, layout, cuphy::tensor_flags::align_tight);
    cuphy::tensor_device deviceTensor(convertToType, layout, tensorDescFlags);

    // Copy the array to pinned memory first.
    size_t nBytes = buf.size * sizeof(T);
    memcpy(hostTensor.addr(), buf.ptr, nBytes);

    // Obtain a tensor in device memory.
    deviceTensor.convert(hostTensor, cuStream);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    return deviceTensor;
}

// Explicit instantiations.
template cuphy::tensor_device deviceFromNumpy<float>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<uint8_t>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<float, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);
template cuphy::tensor_device deviceFromNumpy<std::complex<float>, py::array::f_style | py::array::forcecast>(
    const py::array& py_array,
    cuphyDataType_t convertFromType,
    cuphyDataType_t convertToType,
    cuphy::tensor_flags tensorDescFlags,
    cudaStream_t cuStream);


template <typename T>
T* numpyArrayToPtr(const py::array& py_array) {
    py::array_t<T, py::array::f_style | py::array::forcecast> array = py_array;
    py::buffer_info buf = array.request();
    T* ptr = static_cast<T*>(buf.ptr);
    return ptr;
}

// Explicit instantiations.
template uint8_t* numpyArrayToPtr(const py::array& py_array);
template uint16_t* numpyArrayToPtr(const py::array& py_array);
template uint32_t* numpyArrayToPtr(const py::array& py_array);
template float* numpyArrayToPtr(const py::array& py_array);


template <typename T>
void copyTensorData(cuphy::tensor_ref& tensor, T& tInfo) {
    tInfo.pAddr              = tensor.addr();
    const tensor_desc& tDesc = static_cast<const tensor_desc&>(*(tensor.desc().handle()));
    tInfo.elemType           = tDesc.type();
    std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
}


// Explicit instantiations.
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo1_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo2_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo3_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo4_t& tInfo);
template void copyTensorData(cuphy::tensor_ref& tensor, cuphyTensorInfo5_t& tInfo);

template <typename T>
void copyTensorData(const cuphy::tensor_device& tensor, T& tInfo) {
    tInfo.pAddr              = tensor.addr();
    const tensor_desc& tDesc = static_cast<const tensor_desc&>(*(tensor.desc().handle()));
    tInfo.elemType           = tDesc.type();
    std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
}

// Explicit instantiations.
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo1_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo2_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo3_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo4_t& tInfo);
template void copyTensorData(const cuphy::tensor_device& tensor, cuphyTensorInfo5_t& tInfo);



}  // pycuphy
