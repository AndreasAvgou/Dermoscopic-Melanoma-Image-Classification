## Dermoscopic classification problem in melanoma images

## Problem Statement

Melanoma is a serious form of skin cancer that begins in cells known as melanocytes. While it is less common than basal cell carcinoma (BCC) and squamous cell carcinoma (SCC), melanoma is more dangerous because of its ability to spread to other organs more rapidly if it is not treated at an early stage. The basic examination for melanoma is dermoscopy, an image modality of the skin part that is affected.

## Object

Our object was to create a new deep learning [approach](https://github.com/AndreasAvgou/Dermoscopic-Melanoma-Image-Classification/blob/master/ConvolutionalNeuralNetwork.ipynb), based on convolutional neural networks, to classify dermoscopic images in one out of 32 categories.

## Dataset

You can find the dataset that was used in this [link](https://www.dropbox.com/sh/f506u2n7467em7g/AAB7xlB3Ozsmnyle7OS0FNYaa?dl=0)

## Data Description
```
1) images: contains all the images 
2) meta: contains the meta-data and suggested indexes to use for training/validation/testing.
3) clinic/derm.html: shows all the clinical and dermoscopic images, respectively.
```
## Libraries

All libraries that was used
```
1. Python>=3.8
2. Tensorflow>=2.6.1
3. Keras>=2.4
4. Numpy>=1.19
5. Sklearn>=0.22
6. Matplotlib>=1.19
```
## Cloud Tools

All cloud tool that was used
```
1. Google Drive
2. Google Colab
```
##  Install libraries

How to install all the necessary libraries using pip
```
pip install -r requirements.txt
```
## Citing this work

If you use the code please cite:

Plain Text
```
A. Avgoustis, T. Exarchos, K. L. Kermanidis and P. Mylonas, "Applied Deep learning for categorizing dermoscopic images," 2021 16th International Workshop on Semantic and Social Media Adaptation & Personalization (SMAP), 2021, pp. 1-4, doi: 10.1109/SMAP53521.2021.9610798.
```
BibTex
```bibtex
@INPROCEEDINGS{9610798,
  author={Avgoustis, Andreas and Exarchos, Themis and Kermanidis, Katia Lida and Mylonas, Phivos},
  booktitle={2021 16th International Workshop on Semantic and Social Media Adaptation   Personalization (SMAP)}, 
  title={Applied Deep learning for categorizing dermoscopic images}, 
  year={2021},
  volume={},
  number={},
  pages={1-4},
  doi={10.1109/SMAP53521.2021.9610798}}
```
