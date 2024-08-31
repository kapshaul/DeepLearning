# Mathematical Derivation of Deep Learning

This repository provides a detailed mathematical derivation of deep learning components. The goal is to break down the complex mathematics underlying neural networks for better understanding and application.

### 4. Table of Contents
If the document is long, adding a Table of Contents at the beginning can help readers navigate.

## Table of Contents
1. [Fully Connected Layers](#1-fully-connected-layers)
   
   1.1. [Objective Function and Model Formulation](#11-objective-function-and-model-formulation)

   1.2. [Gradients of Weights and Biases](#12-gradients-of-weights-and-biases)

   1.3. [Backpropagation](#13-backpropagation)
   

## 1. Fully Connected Layers

### 1.1. Objective Function and Model Formulation

A fully connected neural network, which acts as a parametric estimator, can be mathematically expressed as:

$$
\hat{Y} = W_3 \sigma(W_2 \sigma(W_1 X^T + b_1) + b_2) + b_3
$$

Where:
- $\hat{Y}$ represents the predicted output of the network.
- $W_i$ represents the weight matrices associated with each layer $i$.
- $b_i$ enotes the bias vectors corresponding to each layer $i$.
- $\sigma$ denotes the activation function, which introduces non-linearity into the model.
- $X$ represents the input data.

To measure the model’s performance, we use the **Mean Squared Error (MSE)** loss function, defined as:

$$
J = \frac{1}{N} \sum_{i=1}^{N} {(\hat{y}_i - y_i)^2}
$$

The training objective is to minimize the loss function with respect to the weight matrices and bias vectors,

$$
\min_{W_i,b_i}{\frac{1}{N} \sum_{i=1}^{N} {(\hat{y}_i - y_i)^2}}
$$

### 1.2. Gradients of Weights and Biases

To optimize the neural network, we need to compute the gradients of the loss function with respect to each parameter. The gradient of the loss function $J$ with respect to the predicted output $\hat{Y}$:

$\hspace{20pt}\frac{dJ}{d\hat{Y}} = \frac{2}{N} (\hat{Y} - Y)$

Derivation the gradients for the parameters in the final layer, $W_3$ and $b_3$,

$\hspace{20pt}\frac{dJ}{dW_3} = \frac{2}{N} (\hat{Y} - Y) \sigma(W_2 \sigma(W_1 X^T + b_1) + b_2)^T$

$\hspace{20pt}\frac{dJ}{db_3} = \frac{2}{N} \sum_{i=1}^{N}{\hat{y}_i - y_i}$

The gradients with respect to the second layer's parameters, $W_2$ and $b_2$,

$\hspace{20pt}\frac{dJ}{dW_2} = \frac{2}{N} (W_3^T (\hat{Y} - Y)) \circ \sigma\`(W_2 \sigma(W_1 X^T + b_1) + b_2) \sigma(W_1 X^T + b_1)^T$

$\hspace{20pt}\frac{dJ}{db_2} = \frac{2}{N} \sum_{i=1}^{N}((W_3^T (\hat{Y} - Y)) \circ \sigma\`(W_2 \sigma(W_1 X^T + b_1) + b_2))$

Finally, the gradients with respect to the first layer's parameters, $W_1$ and $b_1$, are,

$\hspace{20pt}\frac{dJ}{dW_1} = \frac{2}{N} (W_2^T (W_3^T (\hat{Y} - Y) \circ \sigma\`(W_2 \sigma(W1 X^T + b1) + b2))) \circ \sigma\`(W_1 X^T + b_1) X^T$

$\hspace{20pt}\frac{dJ}{db_1} = \frac{2}{N} \sum_{i=1}^{N}((W_2^T (W_3^T (\hat{Y} - Y) \circ \sigma\`(W_2 \sigma(W_1 X^T + b_1) + b_2))) \circ \sigma\`(W_1 X^T + b_1))$

These calculations can be implemented in MATLAB code as follows,

```MATLAB
% Compute gradients manually
% Gradients for W3 and b3
dW3 = 2/batch_size * (error) * ReLU(W2 * ReLU(W1 * x + b1) + b2)';
db3 = 2/batch_size * sum((error), 2);

% Gradients for W2 and b2
dW2 = 2/batch_size * (W3' * (error)) .* ReLU_deriv(W2 * ReLU(W1 * x + b1) + b2) * ReLU(W1 * x + b1)';
db2 = 2/batch_size * sum((W3' * (error)) .* ReLU_deriv(W2 * ReLU(W1 * x + b1) + b2), 2);

% Gradients for W1 and b1
dW1 = 2/batch_size * (W2' * (W3' * (error) .* ReLU_deriv(W2 * ReLU(W1 * x + b1) + b2))) .* ReLU_deriv(W1 * x + b1) * x';
db1 = 2/batch_size * sum((W2' * (W3' * (error) .* ReLU_deriv(W2 * ReLU(W1 * x + b1) + b2))) .* ReLU_deriv(W1 * x + b1), 2);
```

### 1.3. Backpropagation

To begin, we first backpropagate $\frac{dJ}{d\hat{Y}}$,

$\hspace{20pt}\frac{dJ}{d\hat{Y}} = \frac{2}{N} (\hat{Y} - Y)$

Since $\frac{d\hat{Y}}{dW_3} = \sigma(W_2 \sigma(W_1 X^T + b_1) + b_2)$, by chain rule, the gradients for $W_3$ and $b_3$,

$\hspace{20pt} \frac{dJ}{d\hat{Y}} \frac{d\hat{Y}}{dW_3} = \frac{dJ}{dW_3} = \frac{2}{N} (\hat{Y} - Y) \sigma(W_2 \sigma(W_1 X^T + b_1) + b_2)^T$

$\hspace{20pt}\frac{dJ}{db_3} = \frac{2}{N} \sum_{i=1}^{N}{\hat{y}_i - y_i}$

Backpropagate $\frac{dJ}{dz_2}$ to second hidden layer

Since $\frac{dz_2}{dW_2} = \sigma(W_1 X^T + b_1)$ and $\frac{dJ}{dz_2} = (W_3^T \frac{dJ}{d\hat{Y}}) \circ \sigma\`(W_2 \sigma(W_1 X^T + b_1) + b_2)$, by chain rule, the gradients for $W_2$ and $b_2$,

$\hspace{20pt}\frac{dJ}{dz_2} \frac{dz_2}{dW_2} = \frac{dJ}{dW_2} = \frac{2}{N} (W_3^T (\hat{Y} - Y)) \circ \sigma\`(W_2 \sigma(W_1 X^T + b_1) + b_2) \sigma(W_1 X^T + b_1)^T$

$\hspace{20pt}\frac{dJ}{db_2} = \frac{2}{N} \sum_{i=1}^{N}((W_3^T (\hat{Y} - Y)) \circ \sigma\`(W_2 \sigma(W_1 X^T + b_1) + b_2))$

Backpropagate $\frac{dJ}{dz_1}$ to first hidden layer

Since $\frac{dz_1}{dW_1} = X$ and $\frac{dJ}{dz_1} = (W_2^T \frac{dJ}{dz_2}) \circ \sigma\`(W_1 X^T + b_1)$, by chain rule, the gradients for $W_1$ and $b_1$,

$\hspace{20pt}\frac{dJ}{dz_1} \frac{dz_1}{dW_1} = \frac{dJ}{dW_1} = \frac{2}{N} (W_2^T (W_3^T (\hat{Y} - Y) \circ \sigma\`(W_2 \sigma(W1 X^T + b1) + b2))) \circ \sigma\`(W_1 X^T + b_1) X^T$

$\hspace{20pt}\frac{dJ}{db_1} = \frac{2}{N} \sum_{i=1}^{N}((W_2^T (W_3^T (\hat{Y} - Y) \circ \sigma\`(W_2 \sigma(W_1 X^T + b_1) + b_2))) \circ \sigma\`(W_1 X^T + b_1))$

These expressions can be implemented in MATLAB code as follows,

```MATLAB
% Forward pass
x = X(i:batch_end, :)';
y_true = y(i:batch_end)';

z1 = W1 * x + b1;
a1 = ReLU(z1);

z2 = W2 * a1 + b2;
a2 = ReLU(z2);

z3 = W3 * a2 + b3;
y_pred = z3;

% Compute loss
error =  y_pred - y_true;

% Compute for backpropagation
delta3 = 2/batch_size * (error);

% Gradients for W3 and b3
dW3 = delta3 * a2';
db3 = sum(delta3, 2);

% Backpropagate to second hidden layer
delta2 = (W3' * delta3) .* ReLU_deriv(z2);

% Gradients for W2 and b2
dW2 = delta2 * a1';
db2 = sum(delta2, 2);

% Backpropagate to first hidden layer
delta1 = (W2' * delta2) .* ReLU_deriv(z1);

% Gradients for W1 and b1
dW1 = delta1 * x';
db1 = sum(delta1, 2);
```
