+++
title = 'Understanding Recurrent Neural Networks (RNN)'
date = 2024-10-21T02:41:59+02:00
draft = true
tags = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
categories = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
keywords = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
images = ['/images/20131102_ai_ml_dl/']
+++

## Introduction

Recurrent Neural Networks (RNNs) are a specific type of Neural Networks (NNs) that are especially relevant for sequential data like time series, text, or audio data. In standard NNs all data is treated independently. For example they are not able to capture the temporal relationship in a time series. RNNs however, process the data sequentially to remember data from the past.

## RNN Architecture

In a standard Feedforward Neural Net all data is processed in parallel. As discussed in [Introduction to Deep Learning]({{< ref "/posts/deep_learning/intro_dl.md">}} "intro_dl") it consists of an input layer, an output layer and in between a number of hidden layers. All the outputs are calculated independently and there is no connection beween them. An RNN in contrast uses the output of one step as input of the next step in addition to the input data and in that way creates a connection and a memory to data of previous steps. This way the RNN is able to generate temporal dependencies in a time series or sequential contextdependenciest. The difference in the architecture is illustrated in the following plot.

![rnn_vs_ann](/images/rnn/RNN_vs_ANN.png)
*Illustration of a Recurrent Neural Network vs. a standard Feedforward Artificial Neural Network*

The next plot provides a visualization of one recurrent unit. Shown are the input $x$, the output $o$, and the hidden state $h$. The input state in connected to the hidden state, which is connected  to the output state. Additionally, the output is used as an input, which allows to remember previous information. 

hidden state

![rnn_architecture](/images/rnn/RNN_architecture.png)

< plot NN, RNN folded and unfolded >

## Applications of RNNs

kaggle notebook

## Types of RNNs

* Bidirectural RNN

* GRU

* LSTM

## Challenges of RNNs

## Training RNNs

## Summary

alternative: transformers

kaggle notebook

---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

