# Recurrent Neural Network (RNN)

A Recurrent Neural Network (RNN) processes sequential data by learning patterns over time. It uses feedback connections to retain information from previous steps, enabling it to analyze sequences like text, speech, or time-series data. This ability to capture temporal dependencies makes RNNs ideal for sequentially related tasks.

<div align="center">
    
<img src="https://github.com/kapshaul/DeepLearning/blob/RNN/img/RNN.png" width="600">

**Figure 1**: A Recurrent Neural Network

</div>

#### $\hspace{10pt}$ Why would you prefer a RNN over a Fully Connected Neural Network?

> An RNN is preferable when working with sequential data or time-dependent patterns because of its ability to retain memory of past inputs. This preference arises due to several key advantages:
>
> 1. **Memory of Temporal Dependencies:** RRNNs maintain a hidden state that evolves over time, effectively capturing information from past inputs. This memory feature is crucial for tasks such as language modeling, speech recognition, and time-series forecasting.
>
> 2. **Parameter Efficiency:** RNNs use one set of parameters across all time steps, rather than assigning different parameters for each step in the sequence. This not only controls model size—preventing rapid parameter growth—but also helps RNNs generalize to varying sequence lengths.
> 3. **Sequence Awareness:** RNNs process data in order, one step at a time. This makes them naturally suited to learn the impact of earlier elements on later ones—essential in sequential tasks where context matters.
