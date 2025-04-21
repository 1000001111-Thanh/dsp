/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_UTIL_HPP
#define PYCUPHY_UTIL_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"

namespace py = pybind11;

namespace pycuphy {

template <typename T>
py::array_t<T> deviceToNumpy(uint64_t deviceAddr,
                             uint64_t hostAddr,
                             py::list dimensions,
                             uint64_t cuStream);

template <typename T>
void fromNumpyBitArray(float* src, T* dst, uint32_t npDim0, uint32_t npDim1);


template <typename T>
void toNumpyBitArray(T* src, float* dst, uint32_t dstDim0, uint32_t dstDim1);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0);

template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3);


template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, uint32_t dim0, uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4);

template <typename T>
__attribute__((visibility("default")))
py::array_t<T> hostToNumpy(T* dataPtr, const std::vector<size_t>& dims, const std::vector<size_t>& strides);

template <typename T, int flags = py::array::c_style | py::array::forcecast>
__attribute__((visibility("default")))
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream);

template <typename T, int flags = py::array::c_style | py::array::forcecast>
__attribute__((visibility("default")))
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     void * inputDevPtr,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags hostTensorFlags,
                                     cuphy::tensor_flags deviceTensorFlags,
                                     cudaStream_t cuStream);

template <typename T, int flags = py::array::c_style | py::array::forcecast>
__attribute__((visibility("default")))
cuphy::tensor_device deviceFromNumpy(const py::array& py_array,
                                     cuphyDataType_t convertFromType,
                                     cuphyDataType_t convertToType,
                                     cuphy::tensor_flags tensorDescFlags,
                                     cudaStream_t cuStream);

template <typename T>
T* numpyArrayToPtr(const py::array& py_array);

template <typename T>
void copyTensorData(cuphy::tensor_ref& tensor, T& tInfo);

template <typename T>
void copyTensorData(const cuphy::tensor_device& tensor, T& tInfo);

};

#endif // PYCUPHY_UTIL_HPP
