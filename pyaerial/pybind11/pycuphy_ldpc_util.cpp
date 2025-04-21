/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <stdexcept>

#include "cuphy.h"
#include "util.hpp"
#include "utils.cuh"
#include "pusch_utils.hpp"
#include "pycuphy_ldpc.hpp"


namespace pycuphy {

void printPdschPerTbParams(const PdschPerTbParams& tbParams) {

    printf("\nPdschPerTbParams: \n");
    printf("tbStartAddr:                %p\n", tbParams.tbStartAddr);
    printf("tbStartOffset:              %d\n", tbParams.tbStartOffset);
    printf("tbSize:                     %d\n", tbParams.tbSize);
    printf("cumulativeTbSizePadding:    %d\n", tbParams.cumulativeTbSizePadding);
    printf("testModel:                  %d\n", tbParams.testModel);
    printf("rv:                         %d\n", tbParams.rv);
    printf("Qm:                         %d\n", tbParams.Qm);
    printf("bg:                         %d\n", tbParams.bg);
    printf("Nl:                         %d\n", tbParams.Nl);
    printf("num_CBs:                    %d\n", tbParams.num_CBs);
    printf("Zc:                         %d\n", tbParams.Zc);
    printf("N:                          %d\n", tbParams.N);
    printf("G:                          %d\n", tbParams.G);
    printf("max_REs:                    %d\n", tbParams.max_REs);
    printf("K:                          %d\n", tbParams.K);
    printf("F:                          %d\n", tbParams.F);
    printf("cinit:                      %d\n", tbParams.cinit);
    printf("firstCodeBlockIndex:        %d\n", tbParams.firstCodeBlockIndex);

}

void printPerTbParams(const PerTbParams& tbParams) {

    printf("\nPerTbParams: \n");
    printf("ndi:                    %d\n", tbParams.ndi);
    printf("rv:                     %d\n", tbParams.rv);
    printf("Qm:                     %d\n", tbParams.Qm);
    printf("bg:                     %d\n", tbParams.bg);
    printf("Nl:                     %d\n", tbParams.Nl);
    printf("num_CBs:                %d\n", tbParams.num_CBs);
    printf("Zc:                     %d\n", tbParams.Zc);
    printf("N:                      %d\n", tbParams.N);
    printf("Ncb:                    %d\n", tbParams.Ncb);
    printf("num_CBs:                %d\n", tbParams.num_CBs);
    printf("Ncb_padded:             %d\n", tbParams.Ncb_padded);
    printf("G:                      %d\n", tbParams.G);
    printf("K:                      %d\n", tbParams.K);
    printf("F:                      %d\n", tbParams.F);
    printf("cinit:                  %d\n", tbParams.cinit);
    printf("nDataBytes:             %d\n", tbParams.nDataBytes);
    printf("nZpBitsPerCb:           %d\n", tbParams.nZpBitsPerCb);
    printf("firstCodeBlockIndex:    %d\n", tbParams.firstCodeBlockIndex);
    printf("encodedSize:            %d\n", tbParams.encodedSize);
    printf("layer_map_array:        ");
    for(int layerIdx = 0; layerIdx < tbParams.Nl; layerIdx++) {
        if(layerIdx > 0)
            printf(", ");
        printf("%d", tbParams.layer_map_array[layerIdx]);
    }
    printf("\n");
    printf("userGroupIndex:         %d\n", tbParams.userGroupIndex);
    printf("nBBULayers:             %d\n", tbParams.nBBULayers);
    printf("startLLR:               %d\n", tbParams.startLLR);
    printf("tbSize:                 %d\n", tbParams.tbSize);
    printf("nDmrsCdmGrpsNoData:     %d\n", tbParams.nDmrsCdmGrpsNoData);
}


void setPdschPerTbParams(PdschPerTbParams& tbParams,
                         const uint32_t tbSize,
                         const float codeRate,
                         const uint32_t rateMatchLen,
                         const uint8_t qamMod,
                         const uint32_t numCodeBlocks,
                         const uint32_t numCodedBits,
                         const uint8_t rv,
                         const uint8_t numLayers,
                         const uint32_t cinit) {

    tbParams.tbStartAddr = nullptr;
    tbParams.tbStartOffset = 0;
    tbParams.tbSize = tbSize / 8;
    tbParams.cumulativeTbSizePadding = 0;
    tbParams.testModel =  0;

    tbParams.bg = get_base_graph(codeRate, tbSize);
    tbParams.num_CBs = 0;  // Get this as an output in the following.
    uint16_t Kprime = get_K_prime(tbSize, tbParams.bg, tbParams.num_CBs);
    tbParams.Zc = get_lifting_size(tbSize, tbParams.bg, Kprime);
    tbParams.K = (tbParams.bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * tbParams.Zc;
    tbParams.F = tbParams.K - Kprime;
    tbParams.N = (tbParams.bg == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * tbParams.Zc;
    if(numCodedBits != 0 && tbParams.N != numCodedBits) {  // Just a check as these should match.
        throw std::runtime_error("Invalid number of coded bits!");
    }

    tbParams.Ncb = tbParams.N;
    tbParams.G = rateMatchLen;
    tbParams.Qm = qamMod;
    tbParams.rv = rv;
    tbParams.Nl = numLayers;
    tbParams.max_REs = rateMatchLen / (qamMod * numLayers);
    tbParams.cinit = cinit;
    tbParams.firstCodeBlockIndex = 0;
}


void setPerTbParams(PerTbParams& tbParams,
                    cuphyLDPCParams& ldpcParams,
                    const uint32_t tbSize,
                    const float codeRate,
                    const uint8_t qamMod,
                    const uint32_t ndi,
                    const uint32_t rv,
                    const uint32_t rateMatchLen,
                    const uint32_t cinit,
                    const uint32_t userGroupIdx,
                    const uint8_t numLayers,
                    const uint8_t numUeGrpLayers,
                    const std::vector<uint32_t>& layerMapArray,
                    const uint8_t nDmrsCdmGrpsNoData) {
    tbParams.tbSize = tbSize;  // In bits.
    tbParams.codeRate = codeRate;
    tbParams.Qm = qamMod;
    tbParams.ndi = ndi;
    tbParams.rv = rv;
    tbParams.bg = get_base_graph(codeRate, tbSize);
    tbParams.Nl = numLayers;
    tbParams.num_CBs = 0;
    uint32_t Kprime = get_K_prime(tbSize, tbParams.bg, tbParams.num_CBs);
    tbParams.Zc = get_lifting_size(tbSize, tbParams.bg, Kprime);
    tbParams.N = (tbParams.bg == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * tbParams.Zc;
    tbParams.Ncb = tbParams.N;          // Same as N for now
    tbParams.Ncb_padded = (tbParams.N + 2 * tbParams.Zc + 7) / 8;
    tbParams.Ncb_padded *= 8;
    tbParams.G = rateMatchLen;
    tbParams.K = (tbParams.bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * tbParams.Zc;
    tbParams.F = tbParams.K - Kprime;
    tbParams.cinit = cinit;
    tbParams.nDataBytes = tbSize / 8;
    tbParams.firstCodeBlockIndex = 0;
    tbParams.encodedSize = tbParams.G;
    for(int i = 0; i < numLayers; i++)
        tbParams.layer_map_array[i] = layerMapArray[i];
    tbParams.userGroupIndex = userGroupIdx;
    tbParams.nBBULayers = numUeGrpLayers;
    tbParams.startLLR = 0;

    uint32_t Kd = tbParams.K - tbParams.F - 2 * tbParams.Zc;

    uint32_t numParityNodes;
    uint32_t Zc = tbParams.Zc;
    uint32_t Ncb = tbParams.Ncb;
    uint32_t k0;
    if(tbParams.bg == 1) {

        if(rv == 0) {
            k0 = 0;
        }
        else if(rv == 1) {
            k0 = (17 * Ncb / (66 * Zc)) * Zc;
        }
        else if(rv == 2) {
            k0 = (33 * Ncb / (66 * Zc)) * Zc;
        }
        else if(rv == 3) {
            k0 = (56 * Ncb / (66 * Zc)) * Zc;
        }
        uint32_t NcbForParity = std::min<uint32_t>((tbParams.encodedSize) / tbParams.num_CBs + k0, Ncb);
        numParityNodes = (NcbForParity - Kd + Zc - 1) / Zc;
        numParityNodes = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG1_PARITY_NODES, numParityNodes));
    }
    else {
        if(rv == 0) {
            k0 = 0;
        }
        else if(rv == 1) {
            k0 = (13 * Ncb / (50 * Zc)) * Zc;
        }
        else if(rv == 2) {
            k0 = (25 * Ncb / (50 * Zc)) * Zc;
        }
        else if(rv == 3) {
            k0 = (43 * Ncb / (50 * Zc)) * Zc;
        }
        uint32_t NcbForParity = std::min<uint32_t>((tbParams.encodedSize) / tbParams.num_CBs + k0, Ncb);
        numParityNodes = (NcbForParity - Kd + Zc - 1) / Zc;
        numParityNodes = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG2_PARITY_NODES, numParityNodes));
    }

    ldpcParams.parityNodesArray.push_back(numParityNodes);
    uint32_t Kb = get_Kb(tbSize, tbParams.bg);
    ldpcParams.KbArray.push_back(Kb);

    tbParams.nZpBitsPerCb = (numParityNodes * Zc) + tbParams.K;
    tbParams.mScUciSum = 0;
    tbParams.isDataPresent = 1;
    tbParams.uciOnPuschFlag = 0;
    tbParams.csi2Flag = 0;
    tbParams.debug_d_derateCbsIndices = nullptr;
    tbParams.enableTfPrcd = 0;
    tbParams.nDmrsCdmGrpsNoData = nDmrsCdmGrpsNoData;
}

} // namespace pycuphy
