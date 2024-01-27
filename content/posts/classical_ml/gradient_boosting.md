+++
title = 'Gradient Boost for Regression - Explained'
date = 2024-01-12T09:21:46-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Gradiend Boosting", "Ensemble", "Classification", "Tree Methods", "Regression"]
images = ['/images/']
+++

## Introduction

*Gradient Boosting*, also called *Gradient Boosting Machine (GBM)* is a type of [supervised](supervised_unsupervised.md#supervised) Machine Learning algorithm that is based on [ensemble learning]({{< ref "/posts/ml_concepts/ensemble">}}). It consists of a sequential series of models, each one trying to improve the errors of the previous one. It can be used for both regression and classification tasks. In this post we introduce the algorithm and then explain it in detail for a regression task. We will have a look at the general formulation of the algorithm and then derive and simplify the individual steps for the most common use case, which uses Decision Trees as underlying models and the Squared Error as loss function. Please find a detailed example, where this is applied to a specific dataset in the separate article [Gradient Boosting for Regression - Example](). Gradient Boosting can also be applied for classification tasks. This is covered in the articles [Gradient Boosting for Classification - Explained]() and [Gradient Boosting for Classification - Example]().

## The Algorithm

Gradient Boosting is, as the same suggests, an ensemble model that is based on [boosting]({{< ref "/posts/ml_concepts/ensemble#boosting">}}). In boosting, an initial model is fit to the data. Then a second model is built on the results of the first one, trying to improve the inaccurate results of the first one, and so on until a series of additive models is built, which together are the ensemble model. The individual models are so-called weak learners, which means that they are simple models with low predictive skill, that is only a bit better than random chance. The idea is to combine a set of weak learners to achieve one strong learner, i.e. a model with high predictive skill. 

< IMAGE BOOSTING >

The most popular underlying models in Gradient Boosting are [Decision Trees]({{ ref "/posts/classical_ml/decision_trees">}}), however using other models, is also possible. When a Decision Tree is used as a base model the algorithm is called *Gradient Boosted Trees*, and a shallow tree is used as a weak learner. Gradient Boosting is a [supervised]() Machine Learning algorithm, that means we aim to find a mapping that approximates the target data as good as possible. This is done by minimizing a [Loss Function](), that meassures the error between the true and the predicted values. Common choices for Loss functions in the context of Gradient Boosting are the [Mean Squared Error]() for a regression task and the [logarithmic loss]() for a classification task. It can however be any differentiable function. 

< INTUITIVE EXPLANATION > add residuals


In this section, we will go through the individual steps of the algorithm. For the explanation of the algorithm, we will follow the notations used in [Wikipedia](https://en.m.wikipedia.org/wiki/Gradient_boosting). We will first have a general look at each step of the algorithm and then simplify and explain it for a regression problem with a variation of the [Mean Squared Error]() as the [Loss Function]() and [Decision Trees]() as underlying models. For a concrete example, with all the calculations included for a specific dataset, please check [Gradien Boosting for Regression - Example](). More specifically, we use as a Loss for each sample
$$L(y_i, F(x_i)) = \frac{1}{2}(y_i - F(x_i))^2.$$
The factor $\frac{1}{2}$ is included to make the calculations easier.

Let ${(x_i, y_i)}_i=1^n = {(x_1, y_1), \dots, (x_n, y_n)} be the training data, with $x = x_0, \dots, x_n$  the input features and $y = y_0, \dots, y_n$ the target values and $F(x)$ be the mapping we aim to determine to approximate the target data. The algorithm is then describes as follows.

1. **Make an initial constant prediction.** 

The initial prediction depends on the Loss function ($L$) we choose. Mathematically this initial prediction is defined as 

$$F_0(x) = argmin\lim_{\gamma}\sum_{i=10}^n L(y_i, \gamma)$$, 

where $\gamma$ are the predicted values. For the special case of $L$ being the loss Function defined above, this can be written as 

$$F_0(x) = argmin\lim_{\gamma}\frac{1}{2}\sum_{i=1}^n(y_i - \gamma^2).$$ 

The expression $argmin\lim_{\gamma}$, means that we want to find the value for $\gamma$ that minimizes the equation. To find the minimum, we need to take the derivative with respect to $\gamma$ and set it to zero.

$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \frac{\delta}{\delta \gamma} \sum_{i=1}^n\frac{1}{2}(y_i - \gamma)^2$$
$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = -2 \sum_{i=1}^n \frac{1}{2} (y_i - \gamma)$$
$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = - \sum_{i=1}^n y_i + n\gamma$$

We set this equal to $0$ and get

$$ - \sum_{i=1}^ny_i + n\gamma = 0$$
$$n\gamma = \sum_{i=1}^n y_i$$
$$\gamma = \frac{1}{n}\sum_{i=1}^ny_i = \bar{y}.$$ 

That means for the special loss function we considered, we get the mean of all target values as the first prediction

$$F_0(xx) = \bar{y}.$$

The next steps are repeated $M$ times, with $M$ is the number of estimators or in this special case Decision Trees. We can write the next steps in the form of a loop.

For $m=1$ to $M$:

2A. **Calculate the (pseudo-)residuals of the preditions and the true observations.** 

The (pseudo-)residuals $r_{im}$ are defined as  

$$r_{im} = - \Big[\frac{\delta L(y_i, F(x_i))}{\delta F(x_i)}\Big]_{F(x)=F_{m-1}(x),$$ for $i=1, \dots, n.$

Before simplifying it for the special use case, we are considering, let's have a closer look at this expression. The residuals $r_{im}$ have two indices, the $m$ corresponds to the current model - remember we are building $M$ models. The second index $i$ corresponds to a data sample. That is the residuals are calculated for each sample individually. The right-hand side seems a bit overwhelming, but looking at it more closely, we can see that it is actually only the negative derivative of the Loss Function with respect to the previous prediction. In other words, it is the negative of the Gradient of the Loss Function at the previous iteration. The (pseudo-)residual $r_{im} thus gives the direction and the magnitude to minimize the Loss Function, which shows the relation to [Gradient Descent]().  

Now, let's see what we get, when we use the loss specified above. 

$$r_{im} = -\Big[\frac{\delta L(y_i,F(x_i))}{\delta F(x_i)}\Big]_{F(x)=F_{m-1}(x)}$$ 
$$r_{im} = -\frac{\delta \frac{1}{2}(y_i - F_{m-1})^2}{\delta F_{F_{m-1}}$$
$$r_{im} = (y_i - F_{m-1})$$

That is, for the special Loss $L(x_i, F(x_i)) = \frac{1}{2}(y_i - F(x_i))^2$, the (pseudo-)residuals $r_{im}$, reduce to the difference of the actual target and the predicted value, which is also known as the [residual](). This is also the reason, why the (pseudo-)residual has this name. If we choose a different Loss Function, the expession will change accordingly. 

2B. **Fit a model (Decision Tree) to the residuals.** 

The next step is to train a model with the residuals as target values, that is use the data {(x_i, r_{im})}_{i=1}^m and fit a model to it.

2C. **Calculate improved predictions.** 

The improved predictions are $\hat{y} + \alpha \cdot F_{res}$, with $\alpha$ being the learning rate, which is a hyperparamter between $0$ and $1$ that needs to be chosen. It determines the contribution of each tree. The learning rate $\alpha$ is a parameter that is related with the [Bias-Variance Tradeoff](). A learning rate closer to $1$ usually reduces the bias, but increases the variance and vice versa. That is we choose a lower learning rate to reduce the variance and overfitting.

Repeat 2 and 4 $d$ times.
The final prediction is $\hat{y} + \alpha \cdot r_1 + r_2 + \cdots + \alpha \cdot r_d$, with $r_1, r_2, \dots, r_d$ the predicted residuals from the $d$ weak learner.

< IMAGE FOR GRADIENT BOOSTING MAIN ALGORITHM STEPS > 

< IMAGE FOR GRADIENT BOOSITING FOR REG WITH MSE>

The main difference between these two algorithms is that Gradient boosting has a fixed base estimator i.e., Decision Trees whereas in AdaBoost we can change the base estimator according to our needs.


gradient boosting hekps to reduce the bias

The algorithm was first described by Friedman (1999). 


## Gradient Boosting vs. AdaBoost

## Advantages & Disadvantages Gradient Boosted Trees

**Pros**

* Can deal with missing data and outlier
* Can deal with umerical and categorical data
* flexible, any loss function can be used

**Cons**

## Gradient Boosting in Python

What are default values in sklearn? max_nr_leaves, n_estimators, learning_rate

## Summary

fast & accurate

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

