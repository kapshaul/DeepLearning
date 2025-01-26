# Recurrent Neural Network (RNN)

A Recurrent Neural Network (RNN) processes sequential data by learning patterns over time. It uses feedback connections to retain information from previous steps, enabling it to analyze sequences like text, speech, or time-series data. This ability to capture temporal dependencies makes RNNs ideal for sequentially related tasks.

<div align="center">
    
<img src="https://github.com/kapshaul/DeepLearning/blob/RNN/img/RNN.png" width="700">

**Figure 1**: A Recurrent Neural Network

</div>

#### $\hspace{10pt}$ Why would you prefer a RNN?

> An RNN is preferable when working with sequential data or time-dependent patterns because of its ability to retain memory of past inputs. This preference arises due to several key advantages:
>
> 1. **Memory of Temporal Dependencies:** RNNs maintain a hidden state that evolves over time, effectively capturing information from past inputs. This memory feature is crucial for tasks such as language modeling, speech recognition, and time-series forecasting.
>
> 2. **Parameter Efficiency:** RNNs use one set of parameters across all time steps, rather than assigning different parameters for each step in the sequence. This not only controls model size—preventing rapid parameter growth—but also helps RNNs generalize to varying sequence lengths.
> 3. **Sequence Awareness:** RNNs process data in order, one step at a time. This makes them naturally suited to learn the impact of earlier elements on later ones—essential in sequential tasks where context matters.

Mathematically, RNN can be written as:

$$
h_t = \sigma(W_h h_{t-1} + W_x X^T + b)
$$

# Long-Short Term Memory (LSTM)

Long Short-Term Memory (LSTM) networks are a specialized type of Recurrent Neural Networks (RNNs) designed to efficiently handle both short- and long-term dependencies in sequential data. Using gating mechanisms—input, forget, and output gates—LSTMs regulate how information is stored, updated, and discarded, enabling them to capture long-range relationships more effectively than traditional RNNs.

<div align="center">
    
<img src="https://github.com/kapshaul/DeepLearning/blob/RNN/img/LSTM.png" width="500">

**Figure 2**: A Long Short-Term Memory architecture

</div>

#### $\hspace{10pt}$ Why would you prefer an LSTM?

> An LSTM is preferable when working with sequential data or time-dependent patterns due to its ability to capture both short-term and long-term dependencies while mitigating vanishing gradient issues. This preference is based on several key advantages:
>
> 1. **Memory of Long-Term Dependencies:** LSTMs are designed to retain information over longer periods through their specialized memory cells and gating mechanisms (input, forget, and output gates). This makes them particularly effective for tasks like language modeling, speech recognition, and time-series forecasting, where both recent and distant context are important.
>
> 2. **Vanishing Gradient Mitigation:** The gating mechanisms in LSTMs allow them to selectively learn which information to remember or forget. This structure helps alleviate the vanishing gradient problem commonly faced by standard RNNs, enabling effective training on long sequences.

Mathematically, an LSTM can be expressed as:

```math
\begin{aligned}
    & f_t = \sigma(W_f \cdot [h_{t-1}, X_t] + b_f) & \text{(Forget gate)} \\
    & i_t = \sigma(W_i \cdot [h_{t-1}, X_t] + b_i) & \text{(Input gate)} \\
    & \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, X_t] + b_C) & \text{(Candidate memory)} \\
    & C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t & \text{(New cell state)} \\
    & o_t = \sigma(W_o \cdot [h_{t-1}, X_t] + b_o) & \text{(Output gate)} \\
    & h_t = o_t \odot \tanh(C_t) & \text{(Hidden state output)} \\
\end{aligned}
```
