+++
title = 'Understanding Recurrent Neural Networks (RNN)'
date = 2024-10-21T02:41:59+02:00
draft = true
tags = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
categories = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
keywords = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
images = ['/images/rnn/rnn_architecture.png']
+++

## Introduction

Recurrent Neural Networks (RNNs) are a specific type of Neural Networks (NNs) that are especially relevant for sequential data like time series, text, or audio data. Traditional neural networks process each input independently, meaning they cannot retain information about previous inputs. This makes them ineffective for tasks that require understanding sequences, such as time series forecasting or natural language processing. RNNs however, process the data sequentially, which enables them to remember data from the past.

## RNN Architecture

In a standard NN all data is processed in parallel. As discussed in [Introduction to Deep Learning]({{< ref "/posts/deep_learning/intro_dl.md">}} "intro_dl") they have an input layer, an output layer and in between a set of hidden layers. All the outputs are calculated independently and there is no connection beween them. An RNN in contrast uses the output of one step as input of the next step additionally to the input data and in that way creates a connection and a memory to data of previous steps. The difference is illustrated in the following plot.

![rnn](/images/rnn/rnn_ann.png)
*Feed-Forwrard Neural Net vs. Recurrent Neural Net*

At each time step, the RNN takes in the current input along with the hidden state from the previous step. The hidden state acts as a form of memory, allowing the model to retain information from earlier inputs. The current input may for example be a step in a time series or a word in a sentence and the hidden state provides information from the previous time steps or words in a sentence.  This architecture is known as *Recurrent Unit* and can be seen in the next plot. Both illustrations show an RNN. The one on the right hand side shows the "unfolded" network. In this plot the steps are illustrated after each other, which makes the recurrent nature clearer.

![rnn](/images/rnn/rnn_architecture.png)
*RNN architecture*

Unlike feedforward neural networks, which have distinct weights for each node, RNNs use the same set of weights repeatedly within each layer. That is the number of parameters they need to learn is reduced, making them more efficient for sequential data. In the above plot these are illustrated as $u$ and $w$. However, these shared weights are still updated through [backpropagation]({{< ref "/posts/deep_learning/backpropagation.md">}}) and gradient descent, enabling the network to learn and improve over time.

## The Math behind RNNs

In the above sections we learned how RNN work conceptionally. It processes sequential data by maintaining a hidden state that carries information from previous time steps. Now we will have acloser look at the calculations that need to be done to achieve this.

**1. Hidden State Update**

At each time step $t$, the RNN updates its hidden state using the current input $x_t$ and the previous hidden state $h_{t-1}$

$$h_t = \sigma(W h_{t-1} + U x_{t} + b_h),$$

with $h_t$ the hidden state at time $t$, $x_t$ the input at time $t$, $W$ the weight matrix for the hidden state, $U$ the weight matrix for the input, $b_h$ the bias term, and $\sigma$ a non-linear activation function (e.g. tanh or Sigmoid-function).

**2. Output Calculation**

The output $y_t$ at each step is calculated using the hidden state $h_t$(which depends on the input $x_t$).

$$\hat{y}_t = g(V h_t + b_y),$$

with $y_t$ the output at time $t$, $V$ the weight matrix for the output layer, $b_y$ the bias term, and $g$ an activation function. 
The hidden state $h_t$ is a vector of values that encode past information, but to make meaningful predictions, we need to transform it into a proper output format. This transformation is done using an activation function $g$. The choice of $g$ depends on whether a classification or a regression is considered. 

For a classification the softmax function is used to map the output onto a probability distribution for each class. The softmax function is defined as 

$$g(z_i) = softmax(z_i) = \frac{e^z_i}{\sum_j e^z_j}.$$ 

Using this transformation ensures that the output data lies between $0$ and $1$ and the sum of all outputs sum to 1. Consider this example: We are performing a sentiment analysis of a text and have three sentiment classes (negative, neutral, positive), softmax might produce outputs like (0.1, 0.3, 0.6), meaning there’s a 10% probability for a negative sentiment, 30% of a neutral sentiment and 60% probability that the sentiment is positive. 

For a regression task such a transformation is not necessary. We can skip the activation function and calculate the output as 

$$\hat{y}_t = V h_t + b_y.$$

**3. Backpropagation through time**

To train an RNN, we use a variant of backpropagation called Backpropagation Through Time (BPTT). The loss function $L$ (e.g., mean squared error or cross-entropy loss) is calculated across all time steps

$$L = \sum_{t=1}^T \L(y_t, \hat{y}_t),$$

with $\L$ the loss function, $y_t$ the true label at time step $t$, and $\hat{y}_t$ its prediction.     The gradients of the loss with respect to the weights are computed by unrolling the network over time. Unrolling the RNN means we represent it as a feedforward network, where each time step is treated as a layer. This allows us to apply standard [backpropagation]({{< ref "/posts/deep_learning/backpropagation.md">}}). 

In RNNs, this means computing gradients of the loss with respect to both the output layer weights $V$ and the hidden state weights $W$, since errors propagate backward in time. As a result, each hidden state influences not only the current time step but also multiple previous time steps. 

**4. Vanishing and Exploding Gradients**

One challenge in training RNNs is the vanishing and exploding gradients problem. When backpropagating through many time steps, gradients can shrink exponentially (vanishing gradient) or grow uncontrollably (exploding gradient). If gradients become too small, early time steps barely get updated. On the other hand, if gradients become too large, training becomes unstable.

To mitigate this, techniques like gradient clipping, Long Short-Term Memory (LSTM) networks, or Gated Recurrent Units (GRU) are often used. Explaining these techniques is however not in the scope of this post.

## Types of RNNs

In this paragraph some variants of RNNs are briefly explained.

**Bidirectional RNNs**

Instead of processing data only forward in time, Bidirectional RNNs (BiRNNs) pass information both forward and backward, allowing the network to use future context when making predictions. This is particularly useful for NLP tasks like named entity recognition (NER). When using this variant make sure that the future context will be available, when using the model to make predictions. When for example predicting the weather for the next hours, this is not a model variant you can use.

**Long Short-Term Memory (LSTM)**

LSTMs were developed to overcome issues vanilla RNN showed, especially in the context of vanishing gradients. LSTMs introduce gates (input, forget, and output gates) to control information flow and prevent vanishing gradients. This enables a longer memory than vanilla RNN provide, which makes them more effective for tasks like speech recognition and text generation, where a lot of context is needed.

**Gated Recurrent Units (GRU)**

GRUs were developed after LSTMs and simplify them by combining the forget and input gates into a single update gate, reducing computational cost while still handling long-term dependencies effectively. They are often preferred when training speed is a priority.



https://www.ibm.com/think/topics/recurrent-neural-networks

## RRNs in Python

In this section we will apply what we learned to a simple toy dataset. We develop an RNN in Python and see how we can train it. First, we will code an RNN from scratch to better understand it and then we will see how to use *PyTorch* to do that. To see this code in action check the notebook on [kaggle]

## Applications of RNNs

* kaggle notebook

## Summary

alternative: transformers

kaggle notebook