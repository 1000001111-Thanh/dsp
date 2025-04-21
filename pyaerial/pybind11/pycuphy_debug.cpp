/* Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "cuphy.hpp"
#include "pycuphy_debug.hpp"

namespace pycuphy {

H5DebugDump::H5DebugDump(const std::string& filename) {
    m_hdf5File.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(filename.c_str())));
    hdf5hpp::hdf5_file::open(filename.c_str());
}

H5DebugDump::~H5DebugDump() {
    m_hdf5File->close();
}

void H5DebugDump::dump(const std::string& name, const cuphy::tensor_ref& tensor, cudaStream_t cuStream) {
    cuphy::write_HDF5_dataset(*m_hdf5File, tensor, tensor.desc(), name.c_str(), cuStream);
}

}
