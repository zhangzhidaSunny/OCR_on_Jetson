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


# Jetson Deployment for PaddleOCR



## 1. Prepare Environment

You need to prepare a Jetson development hardware. If you need TensorRT, you need to prepare the TensorRT environment. It is recommended to use TensorRT version 7.1.3;

1. Install PaddlePaddle in Jetson

The PaddlePaddle download [link](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html#python)
Please select the appropriate installation package for your Jetpack version, cuda version, and trt version. Here, we download paddlepaddle_gpu-2.3.0rc0-cp36-cp36m-linux_aarch64.whl.

Install PaddlePaddle：
```shell
pip3 install -U paddlepaddle_gpu-2.3.0rc0-cp36-cp36m-linux_aarch64.whl
```


2. Download PaddleOCR code and install dependencies

Clone the PaddleOCR code:
```
git clone https://github.com/PaddlePaddle/PaddleOCR
```

and install dependencies：
```
cd PaddleOCR
pip3 install -r requirements.txt
```

*Note: Jetson hardware CPU is poor, dependency installation is slow, please wait patiently*

## 2. Perform prediction

Obtain the PPOCR model from the [document](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_en/ppocr_introduction_en.md#6-model-zoo) model library. The following takes the PP-OCRv3 model as an example to introduce the use of the PPOCR model on Jetson:

Download and unzip the PP-OCRv3 models.
```
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xf ch_PP-OCRv3_det_infer.tar
tar xf ch_PP-OCRv3_rec_infer.tar
```

The text detection inference:
```
cd PaddleOCR
python3 tools/infer/predict_det.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/  --image_dir=./doc/imgs/french_0.jpg  --use_gpu=True
```

After executing the command, the predicted information will be printed out in the terminal, and the visualization results will be saved in the `./inference_results/` directory.
![](./images/det_res_french_0.jpg)


The text recognition inference:
```
python3 tools/infer/predict_det.py --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/  --image_dir=./doc/imgs_words/en/word_2.png  --use_gpu=True --rec_image_shape="3,48,320"
```

After executing the command, the predicted information will be printed on the terminal, and the output is as follows:
```
[2022/04/28 15:41:45] root INFO: Predicts of ./doc/imgs_words/en/word_2.png:('yourself', 0.98084533)
```

The text  detection and text recognition inference:

```
python3 tools/infer/predict_system.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/ --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/ --image_dir=./doc/imgs/00057937.jpg --use_gpu=True --rec_image_shape="3,48,320"
```

After executing the command, the predicted information will be printed out in the terminal, and the visualization results will be saved in the `./inference_results/` directory.
![](./images/00057937.jpg)

To enable TRT prediction, you only need to set `--use_tensorrt=True` on the basis of the above command:
```
python3 tools/infer/predict_system.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/ --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/ --image_dir=./doc/imgs/  --rec_image_shape="3,48,320" --use_gpu=True --use_tensorrt=True
```

For more ppocr model predictions, please refer to[document](../../doc/doc_en/models_list_en.md)




# OCR recognition processes.

Divided into 3 models: Det text detection model, cls text direction detection model, rec text recognition model. Support CPU and GPU and TensorRT usage. The models use Paddle's pre-trained models, and use Fastdeploy for model deployment calls, which is convenient and fast. fastdeploy supports CPU inference for ONNXRuntime, GPU inference for TensorRT, and PaddleInference by default.

We use TensorRT's GPU inference. The first run requires generating Paddle's text detection, orientation recognition, and text recognition models as trt files, which is slow and has been uploaded as detailed in the GitHub repository. Generating the model for subsequent inference is faster.
![image](https://user-images.githubusercontent.com/84485935/224459589-77f39ac9-787f-4b6e-8f8b-e2d281ef1372.png)

​


bilibili Video
https://www.bilibili.com/video/BV1KT411a75g/

CSDN Blog
https://blog.csdn.net/m0_46339652/article/details/119875117
