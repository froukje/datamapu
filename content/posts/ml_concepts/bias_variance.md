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

In Machine Learning different error sources exist. Some errors cannot be avoided, for example due to unknown variables in the system analysed. These errors are called *irreducible errors*. On the other hand *reducible errors*, are errors that can be reduced and with that the model's skill can be improved. *Bias* and *Variance* are two of the latter. They are concepts used in supervised Machine Learning to evaluate the model's output compared to the true values. For a Machine Learning model to be generalizable to new unseen data with high predictive skill, it is important that bias and variance are balanced. 

<IMAGE>

## Bias

The *bias* in a Machine Learning model is a systematic error in the predictions due to wrong assumptions during the modelling process. It describes the deviation from the model's prediction to the true target data. Mathematically, the bias is defined as  

$$Bias = E(\hat{Y}) - Y,$$

with $\hat{Y}$ the predictions produced by the Machine Learning model and $Y$ the true target values. That means, the bias is the difference between the expected model predictions and the true values. A bias results from assumptions that are made of the underlying data. Since every Machine Learning model is based on some assumptions, all models underly a certain bias. A **low bias** means that fewer assumptions were made and the model fits the data well. A **high bias** can be introduced by using too simplified assumptions about the mapping that is supposed to be modelled, by e.g. using a model that is too simple. In this case the model is not able to capture the underlying pattern of the data. This is also known as *underfitting*.

### Possibilities to reduce the Bias

In general a low bias is desirable. There is, however, no recipe of how to reduce it. The following methods can be tried. 

**Select a more complex model architecture.** If the selected model is too simple compared to the underlying data, the bias will always be high. For example if a linear model is used to model a non-linear relationship, the model will never be able to capture the underlying pattern, no matter how long and with how much data is trained. 

**Increase the number of input features.** More complexity can not only be introduced by the model structure itself, but also by using more input features. The additional features can help to identify the modelled relationship.

**Gather more training data.** A larger training dataset can help to learn the underlying pattern between input features and target data.

**Decrease regularization.** Regularization techniques are used to prevent overfitting and make the model more generalizable. This is useful if the model shows high variance, as we will see in the next section. However, if the bias is high reducing the regularization may help.

< IMAGE high, low bias >

## Variance

*Variance* is a term from statistics, which measures the spread of a variable around its mean. In Machine Learning it describes the of change in the predictions, when different subsets are used for training, or in other words the variability of the model's prediction. Mathematically the variance described as the expected value of the square of the difference between the predicted values and the expected value of the predictions

$$Variance = E[(\hat{Y} - E[\hat{Y}])^2],$$

with $\hat{Y}$ the predictions produced by the Machine Learning model and $Y$ the true target values. **Low variance** means that the variability between the training on different subsets is low. That is the model is less sensitive to changes in the training data and able to generalize to unseen data equally well independingly of the data subset it was trained on. On the other hand **high variance** means that the model is highly sensitive to the training data and the model results differ depending on the selected subset. High variance implicates that the model fits very well to the training data, but is not able to generalize to new data. This phenomenon is called *overfitting*. High variance can result from a complex model with a large set of features.

### Possibilities to reduce the Variance

To make the model generalizable to new data, a low variance is desirable. As for the bias, there is no recipe to achieve this. The following methods may help to reduce the variance.

**Select a less complex model.** High variance often results from a too complex model, that fits the specific training data sample too well and by doing that oversees the general pattern.

**Use cross validation.** In cross validation the training data is splitted into different subsets that are used to train the model. Tuning the hyperparamters on different subsets can make the model more stable and reduce the variance.

**Select relevant feature.** Analogue to reducing the bias by increasing the number of features, we can try to reduce the variance by removing features and with that reduce the complexity of the model. 

**Use regularization.** Regularization adds an extra term to the loss function, which is used to weight features by their importance.

**Use ensemble models.** [Ensemble learning]({{< ref "ensemble">}}) use multiple of models and aggregate them to one single prediction. Different types of ensemble models exist, [Bagging]({{< ref "ensemble#bagging">}}) is especially suited to reduce the variance. 

< IMAGE high low variance>

< Image Combinations of bias and variance>

* high / high
* high / low
* low / high
* low / low

## Bias-Variance Tradeoff

Concluding the above derivations, it is in general desirable to achieve a low bias aswell as a low variance. This is however not possible, which becomes clear when we consider the general error of a Machine Learning model. Let $Y$ be the true values and $\hat{Y} the model's estimate with $Y = \hat{Y} + \epsilon$ and $\epsilon$ a normally distributed error with mean $0$ and standard deviation $\sigma$. $hat{Y}$ is depending on the dataset the model has been trained on. The expected error, that is aimed to be minimized can then be written as

$$E[(Y - \hat{Y})^2] = E[Y^2 - 2Y\hat{Y} + \hat{Y}^2].$$ 

Due to the linearity of the expected value this can be written as

$$E[(Y - \hat{Y})^2] = E[Y^2] - 2E[Y\hat{Y}] + E[\hat{Y}^2]. (1)$$

Let's consider these three terms individually. We can reformulate $E[Y^2]$ as follows

$$E[Y^2] = E[(\hat{Y} + \epsilon)^2] = E[Y^2] + 2E[Y\epsilon] + E[\epsilon^2].$$

Since the true values $Y$ are independend of the dataset, this is equal to

$$E[Y^2] = Y^2 + 2YE[\epsilon] + E[\epsilon^2].$$

We assumed $\epsilon$ to have mean $0$ and standard deviation $\sigma$ this rewrites to

$$E[Y^2] = Y^2 + \sigma^2. (2)$$

The last term can be written as

$$E[\hat{Y}^2] = E[\hat{Y}^2] - E[\hat{Y}]^2 + E[\hat{Y}]^2 = Var[\hat{Y}] + E[\hat{Y}]^2, (3)$$

with $Var[\hat{Y}]$ the [variance](https://en.wikipedia.org/wiki/Variance) of $\hat{Y}$, which is defined as 

$$Var[\hat{Y}] = E[(\hat{Y} - E[\hat{Y}])^2] = E[\hat{Y}^2 - 2\hat{Y}E[\hat{Y}] + E[\hat{Y}]^2]$$

and can thus be written as

$$Var[\hat{Y}] = E[\hat{Y}^2] - 2E[\hat{Y}]E[\hat{Y}] + E[\hat{Y}]^2 = E[\hat{Y}^2] - E[\hat{Y}]^2.$$

Putting (2) and (3) back into equation (1) leads to

$$E[(Y - \hat{Y})^2)] = Y^2 + \sigma^2 - 2YE[\hat{Y}] + Var[\hat{Y}] + E[\hat{Y}]^2.$$

This can be written as

$$E[(Y - \hat{Y})^2] = (E[\hat[Y] - Y)^2 + Var[\hat{Y}] + \sigma^2.$$

In other words the total error in a Machine Learning model is

$$E[(Y - \hat{Y})^2] = Bias^2 + Variance + \sigma^2,$$

with $\sigma$ being the irreducible error. The total error is thus composed of the Bias, the Variance and the irreducible error.

< IMAGE curves> 

## Summary
