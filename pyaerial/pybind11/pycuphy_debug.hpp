/*
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#ifndef PYCUPHY_DEBUG_HPP
#define PYCUPHY_DEBUG_HPP

#include <string>
#include "cuphy.hpp"
#include "hdf5hpp.hpp"

namespace pycuphy {

class H5DebugDump {

public:
    H5DebugDump(const std::string& filename);
    ~H5DebugDump();

    void dump(const std::string& name, const cuphy::tensor_ref& tensor, cudaStream_t cuStream);

private:
    std::unique_ptr<hdf5hpp::hdf5_file> m_hdf5File;
};

}


#endif // PYCUPHY_DEBUG_HPP
