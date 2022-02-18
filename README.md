# Contextual-Measures

This is an official implementation of "Contextual Measures for Iris Recognition".
The complete source code will come soon.


## Prerequisites
This implementation is based on platform of pytorch 1.7, our environment is:
- Linux
- Python 3.8 (PixelUnshuffle)
- CPU or NVIDIA GPU + CUDA CuDNN
- Pytorch 1.7
- Torchvision  0.8.2
- Pillow  8.1
- Numpy   1.19.5
- Scikit-learn  0.24.0
- Scipy  1.5.4
- Ipython  7.16.1
- Thop (for computational complexity)


## Dataset
Before running the train/test code, it is necessary to 1) download the dataset and 2) configure the data-config file.
### Data download
The CASIA-website ([SIR](http://www.cripacsir.cn/dataset/)) provides 
- the localization results to generate normalized iris images,
- training/testing protocols.

Meanwhile, you can also download normalized iris images of CASIA datasets, including CASIA-iris-V4, CASIA-iris-Mobile, and CASIA Cross Sensor Iris Recognition dataset.


Original iris database:
- CASIA-iris-V4/mobile/cross sensor iris recognition dataset: http://www.cripacsir.cn/dataset/
