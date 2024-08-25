# Solving XOR Problem with Deep Learning

One of the most prominent limitations of a linear model is its inability to solve the XOR problem. While a linear model can handle AND and OR operations, it fails with the XOR operation. In the study of AI, it is commonly known that a Multilayer Perceptron (MLP) can solve the XOR problem. This document explores how this is achieved.

<div align="center">
<img src="https://github.com/kapshaul/DeepLearning/blob/master/XOR/img/Truth-table-XOR-gate.png" width="200">

**Figure 1**: Truth table XOR gate
</div>

## 1. Limitations of a Linear Model

When classifying data, a linear model can only create a linear boundary, effectively dividing data into two categories using a straight line. This approach works for operations like AND and OR, where a linear line can separate the outputs (0 and 1). However, the XOR operation is not linearly separable, meaning a single straight line cannot separate the outputs for XOR.

This limitation of the perceptron was mathematically demonstrated by Marvin Minsky and Seymour Papert in their book *Perceptrons: An Introduction to Computational Geometry* (1969).

<div align="center">
<img src="https://github.com/kapshaul/DeepLearning/blob/master/XOR/img/linear_classification.png" width="500">

**Figure 2**: Linear classification for AND and OR gates, but not for the XOR gate
</div>

## 2. Solving XOR with a Hidden Layer

To solve the XOR problem, at least two lines or a non-linear boundary are required. In this document, we demonstrate how two lines can be used to solve the XOR problem.

### XOR Logic

Conceptually, the XOR operator can be represented by the intersection of two logical operations:

1. **NOT (x1 AND x2)** - A line representing this operation.
2. **(x1 OR x2)** - A line representing this operation.

<div align="center">
<img src="https://github.com/kapshaul/DeepLearning/blob/master/XOR/img/xor_classification.png" width="300">

**Figure 3**: XOR classification with two linear lines
</div>

The intersection of these two lines (AND operation) yields a result of 1 where the XOR condition is met, and 0 otherwise. In the figure, the blue line represents the NAND operation, and the orange line represents the OR operation.

## 3. Classification of XOR with MLP

### Model Architecture

To classify XOR, the input layer must consist of neurons corresponding to the input pairs (0,0), (0,1), (1,0), and (1,1). Therefore, the input layer contains two neurons. The hidden layer needs at least two neurons to form the two lines necessary for classification. These neurons are fully connected (FC) and use weights, biases, and the Sigmoid activation function.

<div align="center">
<img src="https://github.com/kapshaul/DeepLearning/blob/master/XOR/img/xor_nn.png" width="400">

**Figure 3**: Neural network architecture for the XOR classification
</div>

### Activation Function

The Sigmoid function is used as the activation function. It approaches 1 as the input becomes more positive and 0 as the input becomes more negative.

```bash
# Define the neural network model
model = tf.keras.Sequential([
    # Fully connected layer with sigmoid activation
    tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(2,)),
    
    # Output layer with sigmoid activation
    tf.keras.layers.Dense(1, activation="sigmoid")
])
```
