# Using Paddle and PYQT to build OCR Tool on Jetson Xavier NX.
Using Paddle and PYQT to build OCR Tool on Jetson Xavier NX.

We installed Jetpack 4.6.1，CUDA 10.2，TensorRT 8.2，Python 3.7
Software component using PaddlePaddle,Fast-deploy,PyQT.


Our compact AI industrial computer with NVIDIA® Jetson™ Xavier NX series core modules.

NVIDIA® Jetson Xavier™ NX, delivering 21 TOPS of compute power at 20W power consumption mode, with 384 CUDA
Cores, 48 Tensor Cores, 2 NVDLA engines, 6 ARM CPU cores, and 8 GB of 128-bit LPDDR4x 51.2GB/s memory, it can run multiple network models simultaneously.




Python Compilation and Installation
The compilation process also requires that

gcc/g++ >= 5.4 (8.2 recommended)
cmake >= 3.10.0
jetpack >= 4.6.1
python >= 3.6
Python package depends on wheel, please install wheel before compiling

If you need to integrate the Paddle Inference backend, download the corresponding Jetpack C++ package according to your development environment on the Paddle Inference precompiled libraries page and unzip it.

All compilation options are imported via environment variables

git clone https://github.com/PaddlePaddle/FastDeploy.git

cd FastDeploy/python

export BUILD_ON_JETSON=ON

export ENABLE_VISION=ON


##ENABLE_PADDLE_BACKEND & PADDLEINFERENCE_DIRECTORY are optional
export ENABLE_PADDLE_BACKEND=ON

export PADDLEINFERENCE_DIRECTORY=/Download/paddle_inference_jetson


python setup.py build
python setup.py bdist_wheel
The compiled wheel package will be generated in the FastDeploy/python/dist directory when the compilation is done, just pip install it directly
