# Deep Learning Basic

Deep learning is a powerful technique within the broader field of machine learning, where models known as neural networks are used to learn from data. These networks can uncover complex relationships and patterns, making them ideal for tasks such as image recognition, natural language processing, and, in our case, function approximation.

#### $\hspace{10pt}$ What is a Neural Network?

> A neural network is a system of interconnected nodes, or "neurons," arranged in layers. Each neuron receives input, processes it through a series of mathematical operations, and produces an output. This output then serves as input to the next layer in the network, allowing the model to progressively build up more complex representations of the data.

<br>

<div align="center">
    
<img src="https://github.com/kapshaul/DeepLearning/blob/master/img/deep_neural_network.jpg" width="600">

**Figure 1**: A Neural Network

</div>

#### **Layers of the Network**

- **Input Layer:** This is where the model receives its data.
- **Hidden Layers:** These layers perform most of the work in the network. They apply activation functions to the input, transforming it in ways that enable the network to learn.
- **Output Layer:** The final layer produces the network's prediction.

#### **Mathematical  Structure**

A neural network can be mathematically represented using matrix multiplication as follows:

$$
Y = W_2 \sigma(W_1 X^T + b_1) + b_2
$$

where,

- $X \in \mathbb{R}^{n \times m}$ represents the input matrix.
- $Y \in \mathbb{R}^{l \times n}$ represents the output matrix.
- $W_1 \in \mathbb{R}^{k \times m}$ and $W_2 \in \mathbb{R}^{l \times k}$ are weight matrices corresponding to the connections between layers.
- $b_1 \in \mathbb{R}^{k}$ and $b_2 \in \mathbb{R}^{l}$ are bias vectors for the respective layers.
- $\sigma(\cdot)$ denotes the activation function, applied element-wise to the intermediate result.

This formulation captures the essential operations in a simple feedforward neural network with one hidden layer.

## 1. Sine Function Prediction: A Simple Introduction to Deep Learning

This example is designed to provide a basic understanding of deep learning concepts by walking through the process of predicting a sine function using a neural network. We'll explore how to set up a simple model, understand its components, and see it in action using both PyTorch and TensorFlow.

#### $\hspace{10pt}$ Why Sine Function Prediction?

> Predicting the sine function is a classic problem that helps illustrate how neural networks work in a simple and intuitive way. Despite its simplicity, the sine function has enough complexity to demonstrate key deep learning concepts such as activation functions, training loops, and loss minimization.

### 1.1. Key Concepts Explained

#### **Activation Functions: The "If" Statement of Neural Networks**

One of the most important components of a neural network is the activation function. After a neuron computes its weighted sum of inputs, the activation function decides whether that neuron should "fire" (send its signal forward) or not. This is akin to an "if" statement in programming:

- **ReLU (Rectified Linear Unit):** A popular activation function that outputs the input directly if it's positive; otherwise, it outputs zero. This introduces non-linearity, enabling the network to learn from complex data patterns.

- **Sigmoid:** The Sigmoid activation function maps the input to a value between 0 and 1 using the formula $\sigma(x) = \frac{1}{1 + e^{-x}}$. It is often used in the output layer of binary classification models because it can represent a probability. However, it can suffer from the vanishing gradient problem, especially for very large or small input values, which can slow down learning.

- **Tanh (Hyperbolic Tangent):** The Tanh function is similar to the Sigmoid function but maps the input to a value between -1 and 1 using the formula $\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$. It is centered around zero, which makes it often preferred over the Sigmoid function as it tends to produce outputs with stronger gradients. However, like Sigmoid, it can also suffer from the vanishing gradient problem for very large or small input values.

<div align="center">
    
<img src="https://github.com/kapshaul/DeepLearning/blob/master/img/activation%20functions.png" width="600">

**Figure 2**: Activation functions

</div>

### 1.2. Deep Learning Frameworks

#### **PyTorch Model**

The PyTorch implementation defines a custom model class and uses a training loop to optimize the model parameters. The code for this implementation can be found [here](#).

```python
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(1, 50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.fc3 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

#### **TensorFlow Model**

The TensorFlow implementation uses the Sequential API to build the model and the fit method for training. The code for this implementation can be found [here](#).

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])
```

### 1.3. Training

The figure visualizes how the activation function influences the learning process of the neural network when approximating the sine function. The curve shown highlights key points where the network has learned the underlying pattern of the sine wave. These specific points, where the curve changes more noticeably, correspond to activation points where the ReLU function behaves similarly to an "if" statement. In these regions, the ReLU activation either allows the signal to pass through or blocks it, depending on whether the input is positive or negative, thus shaping the curve accordingly.

<div align="center">
    
<img src="https://github.com/kapshaul/DeepLearning/blob/master/img/sinefunction_training_animation.gif" width="600">

**Figure 3**: Training a sine function

</div>
