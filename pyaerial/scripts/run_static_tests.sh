#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Exit on first error
set -e

# Switch to PROJECT_ROOT directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
PROJECT_ROOT=$(dirname $SCRIPT_DIR)
echo $SCRIPT starting...
cd $PROJECT_ROOT

echo -n pyAerial: Static analysis with flake8...
flake8 --config=./scripts/.flake8 --exclude container,external
echo Success.

echo pyAerial: Static analysis with pylint...
pylint --rcfile scripts/.pylintrc ./src/aerial

echo pyAerial: Static type checking...
python3 -m mypy src/aerial \
    --no-incremental \
    --disallow-incomplete-defs \
    --disallow-untyped-defs \
    --no-strict-optional \
    --disable-error-code attr-defined

echo pyAerial: Verify docstring coverage...
pushd src > /dev/null
python3 -m interrogate -vv --omit-covered-files
popd > /dev/null

# Finished
echo $SCRIPT finished with success.
