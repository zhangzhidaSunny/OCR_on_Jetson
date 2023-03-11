# Using Paddle and PYQT to build OCR Tool on Jetson Xavier NX.
Using Paddle and PYQT to build OCR Tool on Jetson Xavier NX.

We installed Jetpack 4.6.1，CUDA 10.2，TensorRT 8.2，Python 3.7
Software component using PaddlePaddle,Fast-deploy,PyQT.


Our compact AI industrial computer with NVIDIA® Jetson™ Xavier NX series core modules.

NVIDIA® Jetson Xavier™ NX, delivering 21 TOPS of compute power at 20W power consumption mode, with 384 CUDA
Cores, 48 Tensor Cores, 2 NVDLA engines, 6 ARM CPU cores, and 8 GB of 128-bit LPDDR4x 51.2GB/s memory, it can run multiple network models simultaneously.

![image](https://user-images.githubusercontent.com/84485935/224459540-8915df09-e6ae-4740-96eb-620d9eae284d.png)



# Python package Compilation and Installation

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

The compiled wheel package will be generated in the FastDeploy/python/dist directory when the compilation is done, just pip install it directly.

# OCR recognition processes.

Divided into 3 models: Det text detection model, cls text direction detection model, rec text recognition model. Support CPU and GPU and TensorRT usage. The models use Paddle's pre-trained models, and use Fastdeploy for model deployment calls, which is convenient and fast. fastdeploy supports CPU inference for ONNXRuntime, GPU inference for TensorRT, and PaddleInference by default.

We use TensorRT's GPU inference. The first run requires generating Paddle's text detection, orientation recognition, and text recognition models as trt files, which is slow and has been uploaded as detailed in the GitHub repository. Generating the model for subsequent inference is faster.
![image](https://user-images.githubusercontent.com/84485935/224459589-77f39ac9-787f-4b6e-8f8b-e2d281ef1372.png)

​
bilibili Video
https://www.bilibili.com/video/BV1KT411a75g/
CSDN Blog
https://blog.csdn.net/m0_46339652/article/details/119875117
