+++
title = 'Bias and Variance'
date = 2024-01-01T09:39:26+01:00
draft = true
featured_image = ''
tags = ["Data Science", "Machine Learning", "Artificial Intelligence"]
categories = ["Data Science", "Machine Learning", "Artificial Intelligence"]
keywords = ["Data Science", "Machine Learning", "Artificial Intelligence"]
+++

## Introduction

In Machine Learning different error sources exist. Some errors cannot be avoided, for example due to unknown variables in the system analysed. These errors are called *irreducible errors*. On the other hand *reducible errors*, are errors that can be reduced, which improves the model's skill. *Bias* and *Variance* are two of the latter. They are concepts used in supervised Machine Learning to evaluate the model's output compared to the true values. For a Machine Learning model to be generalizable to new unseen data with high predictive skill, it is important that bias and variance are balanced. 

<IMAGE>

## Bias

The *bias* in a Machine Learning model is a systematic error in the predictions due to wrong assumptions during the modelling process. It describes the deviation from the model's prediction to the true target data. Mathematically, the bias is defined as  

$$Bias = E(\hat{Y}) - Y,$$

with $\hat{Y}$ the predictions produced by the Machine Learning model and $Y$ the true target values. That means, the bias is the difference between the expected model predictions and the true values. A bias results from assumptions that are made of the underlying data. Since every Machine Learning model is based on some assumptions, all models underly a certain bias. A **low bias** means that fewer assumptions were made and the model fits the data well. A **high bias** can be introduced by using too simplified assumptions about the mapping that is supposed to be modelled, by e.g. using a model that is too simple. In his case the model is not able to capture the underlying pattern of the data. This is also known as *underfitting*.

### Possibilities to reduce the Bias

In general a low bias is desirable. There is, however, no recipe of how to reduce it. The following methods can be tried. 

**Select a more complex model architecture.** If the selected model is too simple compared to the underlying data, the bias will always be high. For example if a linear model is used to model a non-linear relationship, the model will never be able to capture the underlying pattern, no matter how long and with how much data is trained. 

**Increase the number of input features.** More complexity can not only be introduced by the model structure itself, but also by using more input features. The additional features can help to identify the modelled relationship.

**Gather more training data.** A larger training dataset can help to learn the underlying pattern between input features and target data.

**Decrease regularization.** Regularization techniques are used to prevent overfitting and make the model more generalizable. This is useful if the model shows high variance, as we will see in the next section. However, if the bias is high reducing the regularization may help.

< IMAGE high, low bias >

## Variance

*Variance* is a term from statistics, which measures the spread of a variable around its mean. In Machine Learning it describes the of change in the predictions, when a different subset is used for training.

Mathematically the variance can be written as

$$Variance = E[(\hat{Y} - E[\hat{Y}])^2],$$

with $\hat{Y}$ the predictions produced by the Machine Learning model and $Y$ the true target values.

* By the algorithm itself. If an inappropriate algorithm is chosen that is not able to capture the variance of the data.

### Possibilities to reduce the Variance

< IMAGE high low variance>

## Bias-Variance Tradeoff

< IMAGE> 

## Summary
