+++
title = 'Understanding Recurrent Neural Networks (RNN)'
date = 2025-03-21T02:41:59+02:00
draft = false
tags = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
categories = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
keywords = ["Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Neural Nets", "Recurrent Neural Nets", "Time Series"]
images = ['/images/rnn/rnn_architecture.png']
+++

## Introduction

Recurrent Neural Networks (RNNs) are a specific type of Neural Networks (NNs) that are especially relevant for sequential data like time series, text, or audio data. Traditional neural networks process each input independently, meaning they cannot retain information about previous inputs. This makes them ineffective for tasks that require understanding sequences, such as time series forecasting or natural language processing. RNNs however, process the data sequentially, which enables them to remember data from the past.

## RNN Architecture


In a standard Feedforward Neural Net (FFNN) all data is processed in parallel. As discussed in [Introduction to Deep Learning]({{< ref "/posts/deep_learning/intro_dl.md">}} "intro_dl") a FFNN consists of an input layer, an output layer and in between a number of hidden layers. All the outputs are calculated independently and there is no connection between them. An RNN in contrast uses the output of one step as input of the next step in addition to the input data and in that way creates a connection and a memory to data of previous steps. This way the RNN is able to generate temporal dependencies in a time series or sequential context dependencies. The difference in the architecture is illustrated in the following plot.

![rnn_vs_ann](/images/rnn/RNN_vs_ANN.png)
*Illustration of a Recurrent Neural Network vs. a standard Feedforward Artificial Neural Network*

At each time step, the RNN takes in the current input along with the hidden state from the previous step. The hidden state acts as a form of memory, allowing the model to retain information from earlier inputs. For instance, in the context of a time series, the current input could represent a specific time step, and the hidden state would contain information from earlier time steps. Similarly, in a sentence, the current input could be a word, and the hidden state would store information from the words preceding it. This architecture is known as *Recurrent Unit* and can be seen in the next plot. Both illustrations show an RNN. The one on the right hand side shows the "unfolded" network. In this plot the steps are illustrated after each other, which makes the recurrent nature clearer.

![rnn](/images/rnn/rnn_architecture.png)
*RNN architecture*

Unlike FFNNs, which have distinct weights for each node, RNNs use the same set of weights repeatedly within each layer. That is the number of parameters they need to learn is reduced, making them more efficient for sequential data. In the above plot these are illustrated as $u$ and $w$. However, these shared weights are still updated through [backpropagation]({{< ref "/posts/deep_learning/backpropagation.md">}}) and gradient descent, enabling the network to learn and improve over time.

## The Math behind RNNs

In the above sections we learned how RNN work conceptionally. It processes sequential data by maintaining a hidden state that carries information from previous time steps. Now we will have a closer look at the calculations that need to be done to achieve this.

**1. Hidden State Update**

At each time step $t$, the RNN updates its hidden state using the current input $x_t$ and the previous hidden state $h_{t-1}$

$$h_t = \sigma(W h_{t-1} + U x_{t} + b_h),$$

with $h_t$ the hidden state at time $t$, $x_t$ the input at time $t$, $W$ the weight matrix for the hidden state, $U$ the weight matrix for the input, $b_h$ the bias term, and $\sigma$ a non-linear activation function (e.g. tanh or Sigmoid-function).

**2. Output Calculation**

The output $y_t$ at each step is calculated using the hidden state $h_t$ (which depends on the input $x_t$).

$$\hat{y}_t = g(V h_t + b_y),$$

with $y_t$ the output at time $t$, $V$ the weight matrix for the output layer, $b_y$ the bias term, and $g$ an activation function. 
The hidden state $h_t$ is a vector of values that encode past information, but to make meaningful predictions, we need to transform it into a proper output format. This transformation is done using an activation function $g$. The choice of $g$ depends on whether a classification or a regression is considered. 

For any task that needs a probability as an output, the softmax function is used to map the output onto a probability distribution for each class. This might be a classification task or a Natural Language Processing (NLP) task, such as sequence generation (like text generation). In the latter, the output might represent the probability of the next token or word in a sequence rather than a class label. The softmax function is defined as 

$$g(z_i) = softmax(z_i) = \frac{e^z_i}{\sum_j e^z_j}.$$ 

Using this transformation ensures that the output data lies between $0$ and $1$ and the sum of all outputs sum to 1. Consider the following example: We are performing a sentiment analysis of a text and have three sentiment classes (negative, neutral, positive), softmax might produce outputs like (0.1, 0.3, 0.6), meaning there’s a 10% probability for a negative sentiment, 30% of a neutral sentiment and 60% probability that the sentiment is positive. 

For a regression task such a transformation is not necessary. We can skip the activation function and calculate the output as 

$$\hat{y}_t = V h_t + b_y.$$

The equations can be illustrated in the following diagram.

![rnn](/images/rnn/RNN_block.jpg)
*RNN Cell*

**3. Backpropagation through time**

To train an RNN, we use a variant of backpropagation called Backpropagation Through Time (BPTT). The loss function $L$ (e.g., mean squared error or cross-entropy loss) is calculated across all time steps

$$L = \sum_{t=1}^T L(y_t, \hat{y}_t),$$

with $L$ the loss function, $y_t$ the true label at time step $t$, and $\hat{y}_t$ its prediction.     The gradients of the loss with respect to the weights are computed by unrolling the network over time. Unrolling the RNN means we represent it as a feedforward network, where each time step is treated as a layer. This allows us to apply standard [backpropagation]({{< ref "/posts/deep_learning/backpropagation.md">}}). 

In RNNs, this means computing gradients of the loss with respect to both the output layer weights $V$ and the hidden state weights $W$, since errors propagate backward in time. As a result, each hidden state influences not only the current time step but also multiple previous time steps. 

**4. Vanishing and Exploding Gradients**

One challenge in training RNNs is the vanishing and exploding gradients problem. When backpropagating through many time steps, gradients can shrink exponentially (vanishing gradient) or grow uncontrollably (exploding gradient). If gradients become too small, early time steps barely get updated. On the other hand, if gradients become too large, training becomes unstable.

To mitigate this, techniques like gradient clipping, Long Short-Term Memory (LSTM) networks, or Gated Recurrent Units (GRU) are often used. Explaining these techniques is however not in the scope of this post.

## Simple Example

Let's consider a simple example to calculate the forward pass for an RNN.

An RNN cell updates the hidden state ($h_t$) and calculates the output ($\hat{y}_t$). To calculate the next hidden state $h_t$ we need the weights $W$ and $U$, and the bias $b_h$. For the output $\hat{y}$ we need the weight $V$ and the bias $b_y$. Let's consider the following example:

![rnn](/images/rnn/rnn4.png)

![rnn](/images/rnn/rnn6.png)

$$
V = \begin{bmatrix} 0.6 & -0.5 \end{bmatrix}
$$

$$
b_h = \begin{bmatrix} 0.1 & -0.1 \end{bmatrix}, \quad
b_y = 0.2
$$

With the previous hidden state

$$h_{t-1} = \begin{bmatrix} 0.2 & -0.4\end{bmatrix}$$

and the input at time $t$

$$x_{t} = \begin{bmatrix}0.5 & 0.3\end{bmatrix}$$

**Step 1: Compute the hidden state**

In this example we use as activation function $\sigma = \tanh$. We can calculate the hidden state as

$$
h_t = \tanh \left( W h_{t-1} + U x_t + b_h \right),
$$

with the weights defined above, we get

![rnn](/images/rnn/rnn1.png)

![rnn](/images/rnn/rnn2.png)

![rnn](/images/rnn/rnn3.png)

$$
h_t = \tanh \left( \begin{bmatrix} 0.49 \\ 0.09 \end{bmatrix} \right)
$$

$$
h_t = \begin{bmatrix} 0.45 \\ 0.09 \end{bmatrix}
$$

**Step 2: Compute the output**

The output is calculated as

$$
\hat{y}_t = V h_t + b_y,
$$

using the weights defined above, we get

$$
y_t = \begin{bmatrix} 0.6 & -0.5 \end{bmatrix} \begin{bmatrix} 0.45 \\ 0.09 \end{bmatrix} + 0.2
$$

$$
y_t = (0.6 \cdot 0.45) + (-0.5 \cdot 0.09) + 0.2
$$

$$
y_t = 0.27 - 0.045 + 0.2 = 0.425.
$$

Which is the next output.

## Types of RNNs

In this paragraph some variants of RNNs are briefly presented without explaining them in detail.

**Bidirectional RNNs**

Instead of processing data only forward in time, Bidirectional RNNs (BiRNNs) pass information both forward and backward, allowing the network to use future context when making predictions. This is particularly useful for NLP tasks like named entity recognition (NER). When using this variant we need to make sure that the future context will be available, when using the model to make predictions. When for example predicting the weather for the next hours, this is not a model variant we can use.

**Long Short-Term Memory (LSTM)**

LSTMs were developed to overcome issues vanilla RNN showed, especially in the context of vanishing gradients. LSTMs introduce gates (input, forget, and output gates) to control information flow and prevent vanishing gradients. This enables a longer memory than vanilla RNNs provide, which makes them more effective for tasks like speech recognition and text generation, where a lot of context is needed.

**Gated Recurrent Units (GRU)**

GRUs were developed after LSTMs and simplify them by combining the forget and input gates into a single update gate, reducing computational cost while still handling long-term dependencies effectively. They are often preferred when training speed is a priority.

## Input-Output Structures in RNNs

RNNs are versatile and can handle a variety of input-output structures depending on the task. The main types are:

**One-to-One**

This is the simplest structure, where one input produces one output. In this case, the RNN behaves similarly to a feedforward neural network since it does not leverage sequential dependencies across multiple time steps. An example would be time series prediction, where a single time step is used to predict the next time step.

Example: Basic time series forecasting where only the previous value is used to predict the next (though most real-world models use multiple past values, making it many-to-one).

**One-to-Many**

A single input generates a sequence of outputs. This structure is useful in applications where a single source must produce a series of predictions.

Example: Image captioning, where a single image is processed, and the RNN generates a sequence of words to describe it. Typically, a Convolutional Neural Network (CNN) extracts features from the image, which are then fed into an RNN (such as an LSTM) to produce a descriptive caption.

**Many-to-One**

In this structure, a sequence of inputs produces a single output. This is common in tasks that require analyzing an entire sequence before making a final prediction.

Example: Sentiment analysis, where an RNN processes a sequence of words in a customer review and outputs a sentiment classification (e.g., positive, negative, or neutral).

**Many-to-Many**

This structure maps a sequence of inputs to a sequence of outputs. It can either be synchronized (where input and output lengths match) or asynchronous (where the input and output lengths may differ).

Example: Machine translation, where an RNN translates a sentence from one language to another, producing a sequence of words in the target language.

The following table summarizes the different types of input-output structures in RNNs:

![rnn](/images/rnn/RNN_types_tabelle.png)
*RNN types summarized*

The different input and output types can be illustrated as follows.

![rnn](/images/rnn/RNN_types.jpg)
*RNN types illustrated*

By choosing the appropriate RNN architecture and input-output structure, it is possible to tackle a wide range of complex sequence-based problems, from time series prediction to natural language processing.



## RNNs in Python

Finaly, we will create a Recurrent Neural Network (RNN) in Python. For this demonstration, we will use [*PyTorch*](https://pytorch.org/), a widely used deep learning framework. The concepts we've covered so far will be applied to a simple toy dataset.

The training data consists of a basic sequence of numbers, and the task is to predict the next number in the sequence. The following code outlines how to build and train an RNN using PyTorch to accomplish this task. We will implement a basic RNN with a single hidden layer. The RNN model is defined by inheriting from the [*nn.Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) class, and the [*nn.RNN*](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) class is used to create the RNN unit. 

The code appears longer than it actually is due to the detailed comments, which are provided within the code for further clarification. The code snipped although a very simplified dataset, is a full working example.

```Python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate a simple sequence dataset
n = 10  # Length of the sequence
# X_train is the input sequence, which is a range of numbers from 0 to n. 
# It is reshaped to be of shape [n+1, 1, 1] to represent a sequence of 1 step at a time.
X_train = torch.arange(n+1, dtype=torch.float32).view(-1, 1, 1)  # Shape: [n+1, 1, 1]
# y_train is the target sequence, which is the next number in the sequence.
# It is reshaped similarly to X_train.
y_train = torch.arange(1, n+2, dtype=torch.float32).view(-1, 1, 1)  # Shape: [n+1, 1, 1]

# Define RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # The RNN layer expects input of shape (batch_size, seq_length, input_size)
        # The hidden size is the number of neurons in the hidden layer
        # batch_first=True ensures that the input/output will have shape 
        # (batch_size, seq_length, features)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  
        # A fully connected layer that takes the hidden state output and maps it to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize the hidden state to zeros 
        # (1 layer, 
        # batch size x sequence length, i.e. the entire length of the sequence for this simple example, 
        # hidden size)
        h0 = torch.zeros(1, x.size(0), hidden_size)  
        # Pass the input through the RNN layer
        out, _ = self.rnn(x, h0)  # 'out' is the output from the RNN
        # Pass the RNN output through the fully connected layer to generate the final output
        out = self.fc(out)
        return out

# Define the hidden layer size
# The hidden size is the number of neurons in the hidden layer
# It is set to 5 in this example, 
# usually determined through trial and error or based on problem complexity
hidden_size = 5
# Initialize the model with input size 1 (since each input is a single number), 
# hidden size 5, and output size 1 (predicting a single value)
model = SimpleRNN(input_size=1, hidden_size=hidden_size, output_size=1)

# Loss function (Mean Squared Error) 
# to measure the difference between predicted and actual values
criterion = nn.MSELoss()
# Optimizer (Adam) to update the model's weights based on the computed gradients
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    # Zero the gradients for the optimizer to prevent accumulation
    optimizer.zero_grad()
    # Forward pass: Compute model's output given the input sequence
    output = model(X_train)
    # Compute the loss between predicted output and actual target
    loss = criterion(output, y_train)
    # Backward pass: 
    # Compute the gradients of the loss with respect to the model's parameters
    loss.backward()
    # Update the model's parameters using the optimizer and computed gradients
    optimizer.step()

    # Print the loss every 100 epochs to monitor training progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

## Summary

In summary, RNNs are powerful tools for handling sequential data. They are a class of artificial neural networks designed to process sequences by maintaining a memory of previous inputs. Unlike traditional feedforward networks, RNNs have loops that allow information to persist, making them suitable for tasks where context from earlier steps is essential. While effective for many applications, RNNs face challenges like vanishing gradients, which hinder learning long-term dependencies. Variants such as Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) address these issues with improved memory mechanisms.

Transformers have emerged as a dominant alternative to RNNs. By using a self-attention mechanism instead of sequential processing, transformers can model long-range dependencies more efficiently and support parallel computation. This makes them faster to train and more effective for tasks like natural language processing, where they have largely replaced RNN-based models. This is however out of the scope of this article.


---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

