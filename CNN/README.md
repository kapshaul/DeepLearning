# Convolutional Neural Network (CNN)

A Convolutional Neural Network (CNN) analyzes images by learning filters that detect features like edges and textures. These filters build increasingly complex patterns layer by layer, enabling CNNs to recognize objects and understand spatial relationships, making them ideal for image-related tasks.

<div align="center">
    
<img src="https://github.com/kapshaul/DeepLearning/blob/CNN/img/CNN.jpeg" width="700">

**Figure 1**: A Convolutional Neural Network

</div>

#### $\hspace{10pt}$ Why would you prefer a CNN over a Fully Connected Neural Network?

> A CNN is preferable over a Fully Connected Neural Network when working with image data or other types of spatially structured data. This preference stems from several key advantages:
>
> 1. **Parameter Efficiency:** CNNs significantly reduce the number of parameters by sharing weights across spatial dimensions, making them more efficient and less prone to overfitting, especially in large-scale image processing tasks.
>
> 2. **Feature Learning:** CNNs automatically learn hierarchical feature representations from raw data, starting from low-level features like edges to high-level features like objects.
>
> 3. **Translational Invariance:** Through techniques like pooling and convolution, CNNs can recognize patterns regardless of their position in the input, making them highly effective for tasks like object detection and image classification.

## Key Concepts

### 1. Filter

A filter, or kernel, is a small matrix of weights in a CNN that detects specific characteristics within the input data, such as edges, textures, or patterns. During the convolution operation, the filter slides across the input (e.g., an image), performing element-wise multiplication and summing the results. Each filter is designed to capture unique features from the image.
**Figure 3** illustrates how six different filters extract distinct characteristics from the same image.

<div align="center">

<img src="https://github.com/kapshaul/DeepLearning/blob/CNN/img/filter1.png" width="400">

<img src="https://github.com/kapshaul/DeepLearning/blob/CNN/img/filter2.png" width="400">

**Figure 2**: CNN Filter

</div>

<div align="center">

<img src="https://github.com/kapshaul/DeepLearning/blob/CNN/img/multi-filter.png" width="500">

**Figure 3**: CNN Filters Producing Different Outputs

</div>

### 2. Stride

Stride refers to the step size with which the convolution filter moves across the input data. A stride of 1 means the filter moves one pixel at a time, while a stride of 2 or more means the filter skips pixels as it moves. Larger strides reduce the spatial dimensions of the output but may result in a loss of information.

</div>

<div align="center">

<img src="https://github.com/kapshaul/DeepLearning/blob/CNN/img/stride.png" width="400">

**Figure 4**: Convolutional Stride

</div>

### 3. Padding

Padding involves adding extra pixels around the input data to control the spatial dimensions of the output. This is often done to preserve the original size of the input after convolution.

</div>

<div align="center">

<img src="https://github.com/kapshaul/DeepLearning/blob/CNN/img/Padding.png" width="400">

**Figure 5**: Padding the Input Data Before Applying Filters

</div>

### 4. Pooling

Pooling is a downsampling operation that reduces the spatial dimensions of the input volume, thereby decreasing the computational load and helping to achieve spatial invariance.

</div>

<div align="center">

<img src="https://github.com/kapshaul/DeepLearning/blob/CNN/img/pooling.png" width="400">

**Figure 6**: Pooling Layers

</div>

<!--

## Table of Contents
- [1. MNIST](#1-mnist)

---

## 1. MNIST
-->
