#!/bin/bash

if [ ! -d "venv" ]; then
	python3 -m venv venv
	source ./venv/bin/activate
else
	source ./venv/bin/activate
fi

# install llama-cpp-python with cuBLAS
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
# pip install llama-cpp-python -C cmake.args="-DLLAMA_CUBLAS=on"

pip install -r requirements.txt
