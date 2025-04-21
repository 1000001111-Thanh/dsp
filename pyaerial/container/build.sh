#!/bin/bash -e
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Usage:
# AERIAL_BASE_IMAGE=<base image> $cuBB_SDK/pyaerial/container/build.sh

# Switch to SCRIPT_DIR directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
echo $SCRIPT starting...
cd $SCRIPT_DIR

source ../../cuPHY-CP/container/setup.sh

AERIAL_PLATFORM=${AERIAL_PLATFORM:-amd64}
TARGETARCH=$(basename $AERIAL_PLATFORM)
case "$TARGETARCH" in
    "amd64")
        CPU_TARGET=x86_64
        ;;
    "arm64")
        CPU_TARGET=aarch64
        ;;
    *)
        echo "Unsupported target architecture"
        exit 1
        ;;
esac

# Base image repository and version.
if [[ -z $AERIAL_BASE_IMAGE ]]; then
    AERIAL_BASE_IMAGE=${AERIAL_REPO}${AERIAL_IMAGE_NAME}:${AERIAL_VERSION_TAG}
fi

# Target image name.
PYAERIAL_IMAGE=${PYAERIAL_IMAGE:-pyaerial:$USER-${AERIAL_VERSION_TAG}}


hpccm --recipe pyaerial_recipe.py --cpu-target $CPU_TARGET --format docker --userarg AERIAL_BASE_IMAGE=$AERIAL_BASE_IMAGE BUILD_PYAERIAL=1 > Dockerfile_tmp
if [[ -n "$AERIAL_BUILDER" ]]
then
    docker buildx build --builder $AERIAL_BUILDER --pull --push --platform $AERIAL_PLATFORM --build-arg BUILD_ARGS="$BUILD_ARGS" -t ${PYAERIAL_IMAGE}-${TARGETARCH} -f Dockerfile_tmp .
else
    BUILD_ARGS=${BUILD_ARGS:-"-DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native"}
    DOCKER_BUILDKIT=1 docker build --network host --platform $AERIAL_PLATFORM --build-arg BUILD_ARGS="$BUILD_ARGS" -t ${PYAERIAL_IMAGE}-${TARGETARCH} -f Dockerfile_tmp .
fi
rm Dockerfile_tmp
