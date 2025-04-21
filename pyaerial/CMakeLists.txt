# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(pyaerial LANGUAGES C CXX ASM)

enable_language(CUDA)

# This is by default ON in cuPHY too.
option(ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR "Enable PUSCH per-UE pre-EQ noise-interference power reporting" ON)

if (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=1)
else (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)
    add_definitions(-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR=0)
endif (ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})

set(SOURCES_CPP pybind11/pycuphy_srs_chest.cpp
                pybind11/pycuphy_channel_est.cpp
                pybind11/pycuphy_channel_eq.cpp
                pybind11/pycuphy_noise_intf_est.cpp
                pybind11/pycuphy_cfo_ta_est.cpp
                pybind11/pycuphy_ldpc_decoder.cpp
                pybind11/pycuphy_ldpc_derate_match.cpp
                pybind11/pycuphy_rsrp.cpp
                pybind11/pycuphy_pusch_params.cpp
                pybind11/pycuphy_ldpc_util.cpp
                pybind11/pycuphy_crc_check.cpp
                pybind11/pycuphy_util.cpp
                pybind11/pycuphy_debug.cpp
                pybind11/pycuphy_chan_model.cpp)
add_library(pycuphycpp ${SOURCES_CPP})
target_link_libraries(pycuphycpp PUBLIC cuphy_channels cuphy chanModels ${Python3_LIBRARIES})
target_include_directories(pycuphycpp PRIVATE ../cuPHY/examples/common
                                              ./external/pybind11/include
                                              ../testBenches/chanModels/src)

set(SOURCES pybind11/pycuphy_pybind.cpp
            pybind11/pycuphy_pdsch.cpp
            pybind11/pycuphy_pusch.cpp
            pybind11/pycuphy_csirs_tx.cpp
            pybind11/pycuphy_util.cpp
            pybind11/pycuphy_pdsch_params.cpp
            pybind11/pycuphy_pusch_params.cpp
            pybind11/pycuphy_channel_est.cpp
            pybind11/pycuphy_channel_eq.cpp
            pybind11/pycuphy_noise_intf_est.cpp
            pybind11/pycuphy_cfo_ta_est.cpp
            pybind11/pycuphy_srs_chest.cpp
            pybind11/pycuphy_ldpc_decoder.cpp
            pybind11/pycuphy_ldpc_encoder.cpp
            pybind11/pycuphy_ldpc_rate_match.cpp
            pybind11/pycuphy_ldpc_derate_match.cpp
            pybind11/pycuphy_ldpc_util.cpp
            pybind11/pycuphy_crc_check.cpp
            pybind11/pycuphy_trt_engine.cpp
            pybind11/pycuphy_rsrp.cpp
            pybind11/pycuphy_chan_model.cpp
            pybind11/pycuphy_debug.cpp
            ../cuPHY/src/cuphy/tensor_desc.cpp
)

add_subdirectory(tests/cpp)
add_subdirectory(external/pybind11)


pybind11_add_module(_pycuphy ${SOURCES})
target_link_libraries(_pycuphy PUBLIC cuphy_channels cuphy chanModels)
target_include_directories(_pycuphy PRIVATE ../cuPHY/examples/common ../testBenches/chanModels/src)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_pycuphy PUBLIC VERSION_INFO=${EXAMPLE_VERSION_INFO})

# Copy the necessary libraries to the output directory for pip install
function(copy_to_pycuphy_output_dir tgt)
    add_custom_command(TARGET _pycuphy POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:${tgt}>" "${CMAKE_CURRENT_SOURCE_DIR}/src/aerial/pycuphy/"
    )
endfunction()

# Remove any existing dynamic libraries from pycuphy output directory
# to avoid confusion if building with static libraries only
function(clean_up_pycuphy_output_dir)
    add_custom_command(TARGET _pycuphy POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E echo "Cleaning up pyaerial output directory"
        && "${CMAKE_COMMAND}" -E remove -f "${CMAKE_CURRENT_SOURCE_DIR}/src/aerial/pycuphy/*.so*")
endfunction()

clean_up_pycuphy_output_dir()
copy_to_pycuphy_output_dir(_pycuphy)

if (BUILD_SHARED_LIBS)
    copy_to_pycuphy_output_dir(cuphy)
    copy_to_pycuphy_output_dir(nvlog)
    copy_to_pycuphy_output_dir(fmtlog-shared)
    copy_to_pycuphy_output_dir(chanModels)
endif()
