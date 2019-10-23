#!/bin/bash

git submodule init
git submodule update
pip install -r requirements.txt

function install {
    cd $1
    pip install --upgrade --no-deps --force-reinstall -e .
    cd ..
}

install arglib
install trainlib
install devlib

pip install --verbose --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

