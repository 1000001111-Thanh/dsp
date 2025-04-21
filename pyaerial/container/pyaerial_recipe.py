# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""PyAerial release Docker image hpccm recipe using Aerial release image as the base.

Usage:
$ hpccm
    --recipe pyaerial_recipe.py
    --format docker
    --userarg AERIAL_BASE_IMAGE
    --userarg BUILD_PYAERIAL (0/1)
"""
BUILD_PYAERIAL = int(USERARG.get("BUILD_PYAERIAL"))
AERIAL_BASE_IMAGE = USERARG.get("AERIAL_BASE_IMAGE")
if AERIAL_BASE_IMAGE is None:
    raise RuntimeError("Environment variable AERIAL_BASE_IMAGE must be set")

Stage0 += baseimage(image=AERIAL_BASE_IMAGE, _distro='ubuntu22', _arch=cpu_target)

# Add PyAerial specific requirements.
Stage0 += user(user='root')

if cpu_target == 'x86_64':
    TARGETARCH='amd64'
elif cpu_target == 'aarch64':
    TARGETARCH='arm64'
else:
    raise RuntimeError("Unsupported platform")

Stage0 += shell(commands=[
    f"wget -q https://github.com/jgm/pandoc/releases/download/3.1.11.1/pandoc-3.1.11.1-1-{TARGETARCH}.deb",
    f"dpkg -i pandoc-3.1.11.1-1-{TARGETARCH}.deb",
    f"rm pandoc-3.1.11.1-1-{TARGETARCH}.deb"
    ])

Stage0 += pip(requirements=f'requirements-{TARGETARCH}.txt', pip='pip3')

if cpu_target == 'x86_64':  # To be able to install PyTorch together with TF. For now just x86.
    Stage0 += pip(args=['uninstall nvidia-cudnn-cu12'], pip='pip3')
    Stage0 += pip(packages=['nvidia-cudnn-cu12==8.9.7.29'], pip='pip3')

Stage0 += user(user='aerial')

Stage0 += raw(docker='ARG BUILD_ARGS=')

# Build and install PyAerial.
if BUILD_PYAERIAL:
    Stage0 += shell(commands=[
        'cmake -Bbuild -GNinja --log-level=warning -DNVIPC_FMTLOG_ENABLE=OFF ${BUILD_ARGS}',
        'cmake --build build -t _pycuphy pycuphycpp',
        './pyaerial/scripts/install_dev_pkg.sh',
        ])

Stage0 += workdir(directory='/home/aerial')
Stage0 += raw(docker='CMD /bin/bash')
