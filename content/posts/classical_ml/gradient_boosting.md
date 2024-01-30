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

In this section, we will go through the individual steps of the algorithm in detail. The algorithm was first described by Friedman (1999)[1]. 
For the explanation, we will follow the notations used in [Wikipedia](https://en.m.wikipedia.org/wiki/Gradient_boosting). The next plot shows the very general formulation of Gradient Boosting following [Wikipedia](https://en.m.wikipedia.org/wiki/Gradient_boosting)

< IMAGE FOR GRADIENT BOOSTING MAIN ALGORITHM STEPS > 


We will now have a look at each single step. First, we will explain the general formulation and then modify and simplify it for a regression problem with a variation of the [Mean Squared Error]() as the [Loss Function]() and [Decision Trees]() as underlying models. More specifically, we use as a Loss for each sample
$$L(y_i, F(x_i)) = \frac{1}{2}(y_i - F(x_i))^2.$$
The factor $\frac{1}{2}$ is included to make the calculations easier. For a concrete example, with all the calculations included for a specific dataset, please check [Gradient Boosting for Regression - Example](). 

Let ${(x_i, y_i)}_{i=1}^n = {(x_1, y_1), \dots, (x_n, y_n)}$ be the training data, with $x = x_0, \dots, x_n$  the input features and $y = y_0, \dots, y_n$ the target values and $F(x)$ be the mapping we aim to determine to approximate the target data. Let's start with the first step of the algorithm defined above.

**Step 1 - Initialize the model with a constant value ($F_0(x)$).** 

The initial prediction depends on the Loss function ($L$) we choose. Mathematically this initial prediction is defined as 
$$F_0(x) = \underset{\gamma}{\text{argmin}}\sum_{i=1}^n L(y_i, \gamma), $$

where $\gamma$ are the predicted values. For the special case that $L$ is the loss Function defined above, this can be written as 

$$F_0(x) = \underset{\gamma}{\text{argmin}}\frac{1}{2}\sum_{i=1}^n(y_i - \gamma)^2.$$ 

The expression $\underset{\gamma}{\textit{argmin}}$, means that we want to find the value for $\gamma$ that minimizes the equation. To find the minimum, we need to take the derivative with respect to $\gamma$ and set it to zero.

$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = \frac{\delta}{\delta \gamma} \sum_{i=1}^n\frac{1}{2}(y_i - \gamma)^2$$
$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = -2 \sum_{i=1}^n \frac{1}{2} (y_i - \gamma)$$
$$\frac{\delta}{\delta \gamma} \sum_{i=1}^n L = - \sum_{i=1}^n y_i + n\gamma$$

We set this equal to $0$ and get

$$ - \sum_{i=1}^ny_i + n\gamma = 0$$
$$n\gamma = \sum_{i=1}^n y_i$$
$$\gamma = \frac{1}{n}\sum_{i=1}^ny_i = \bar{y}.$$ 

That means for the special Loss Function we considered, we get the mean of all target values as the first prediction

$$F_0(x) = \bar{y}.$$

The next steps are repeated $M$ times, with $M$ is the number of weak learners or for the special case considered, Decision Trees. We can write the next steps in the form of a loop.

**Step 2 - For $m=1$ to $M$:**

**2A. Compute the (pseudo-)residuals of the preditions and the true observations.** 

The (pseudo-)residuals $r_{im}$ are defined as  

$$r_{im} = -\Big[\frac{\delta L(y_i, F(x_i)}{\delta F(x_i))} \Big]_{F(x)} $$
$$r_{im} = - \Big[\frac{\delta L(y_i, F(x_i))}{\delta F(x_i)}\Big]_{F(x)=F_{m-1}(x)},$$ for $i=1, \dots, n. (1a)$

Before simplifying it for the special use case, we are considering, let's have a closer look at this expression. The residuals $r_{im}$ have two indices, the $m$ corresponds to the current model - remember we are building $M$ models. The second index $i$ corresponds to a data sample. That is the residuals are calculated for each sample individually. The right-hand side seems a bit overwhelming, but looking at it more closely, we can see that it is actually only the negative derivative of the Loss Function with respect to the previous prediction. In other words, it is the negative of the Gradient of the Loss Function at the previous iteration. The (pseudo-)residual $r_{im} thus gives the direction and the magnitude to minimize the Loss Function, which shows the relation to [Gradient Descent]().  

Now, let's see what we get, when we use the loss specified above. 

$$r_{im} = -\Big[\frac{\delta L(y_i,F(x_i))}{\delta F(x_i)}\Big]_{F(x)=F_{m-1}(x)}$$ 
$$r_{im} = -\frac{\delta \frac{1}{2}(y_i - F_{m-1})^2}{\delta F_{F_{m-1}}$$
$$r_{im} = (y_i - F_{m-1}) (1b)$$

That is, for the special Loss $L(x_i, F(x_i)) = \frac{1}{2}(y_i - F(x_i))^2$, the (pseudo-)residuals $r_{im}$, reduce to the difference of the actual target and the predicted value, which is also known as the [residual](). This is also the reason, why the (pseudo-)residual has this name. If we choose a different Loss Function, the expession will change accordingly. 

**2B. Fit a model (weak learner) closed under scaling $h_m(x)$ to the residuals.** 

The next step is to train a model with the residuals as target values, that is use the data {(x_i, r_{im})}_{i=1}^m and fit a model to it. For the special case discussed we train a Decision Tree with a restricted number of leaves or restricted number of depth.

**2C. Find optimized solution $\gamma_m$ for the Loss Function.**

The general formulation of this step is described by solving the optimization problem

$$\gamma_m = \argmin\lim{\gamma}\sum_{i=1}^nL(y_i, F_{m-1}(x_i) + \gamma h_m(x_i)), (2a)$$

where $h_m(x_i)$ is the just fitted model (weak learner) at $x_i$. For the case of using Decision Trees as a weak learner, $h(x_i)$ is

$$h(x_i) = \sum_{j=1}^{J_m} b_{jm} 1_{R_{jm}}(x),$$

with $J_m$ the number of leaves or terminal nodes of the tree, and $R_{1m}, \dots R_{J_{m}m}$ are so-called *regions*. These regions refer to the terminal nodes of the Decision Tree. Because we are fitting a weak learner, that is a prined tree, the terminal nodes will consist of several predictions. Each region relates to one constant prediction, which is the mean over all values in the according node and is denoted as $b_{jm}$ in the above equation. The notations may seem a bit complicated, but once illustated, they should become more clear. An overview is givem in the below plot.

<IMAGE WITH NOTATION FOR A DECISION TREE> R_jm, etc,

For a Decision Tree as underlying model, this step is a bit modifed. A separate optimal value $\gamma_{jm}$ for each of the tree's regions is chosen, instead of a single $\gamma_{m}$ for the whole tree [1, 2]. The coefficients $b_{jm}$ can be then discarded and the equation (2a) is reformulated as

$$\gamma_m = \argmin\lim{\gamma}\sum_{x_i \in{R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma). (2b)$$

Note, that the sum only goes over the elements of the region, which simplifies the notation a bit. Using the specified Loss $L(y_i, F_{m-1}(x_i)) = \frac{1}{2}(y_i - F_{m-1}(x_i))^2$, this reduces to

$$\gamma_m = \argmin\lim{\gamma}\sum_{x_i \isin{R_{jm}} \frac{1}{2}(y_i - (F_{m-1}(x_i) + \gamma))^2.$$

As explained above, this means that we want to minimize the right-hand term. For that we calculate the derivative with respect to $\gamma$ and set it to zero.

$$\frac{\delta}{\delta \gamma}\sum_{x_i\in R_{jm}} \frac{1}{2}(y_i - F_{m-1}(x_i) - \gamma)^2 = 0$$
$$-\sum_{x_i \in R__{jm}} (y_i - F_{m-1}(x) - \gamma) = 0$$
$$-n_j \gamma = \sum_{x_i\in R_{jm}}(y_i - F_{m-1}(x_i)),$$

with $n_j$ the number of samples in the terminal node $R_{jm}$. This leads to

$$\gamma = \frac{1}{n_j}\sum_{x_i\inR_{jm}r_{im}, (2c)$$

with $r_{im} = y_i - F_{m-1}(x_i)$ the residual. The solution that minimizes (2b) is thus the mean over all target values of the tree, we constructed using the residuals as target values. That is $\gamma$ is nothing but the prediction we get from our tree fitted to the residuals.

**2D. Update the model.** 

The last step in this loop is to update the model.

$$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$ 

That is we use our previous model $F_{m-1}$ and add the new predictions from the model fitted to the residuals. For the special case of Decision Trees as weak learners, this can be reformulated to

$$F_{m}(x) = F_{m-1}(x) + \alpha \sum_{j=1}^{J_m} \gamma_{jm}1(x\inR_{jm}).$$

The sum means, that we sum all values $\gamma_{jm}$ of the terminal node $R_{jm}.$ The factor $\alpha$ is the learning rate, which is a hyperparamter between $0$ and $1$ that needs to be chosen. It determines the contribution of each tree and is also often refered to as scaling of the models. The learning rate $\alpha$ is a parameter that is related with the [Bias-Variance Tradeoff](). A learning rate closer to $1$ usually reduces the bias, but increases the variance and vice versa. That is we choose a lower learning rate to reduce the variance and overfitting.

**Step 3 - Output final model $F_M(x)$.**

The individual steps of algorithm for the special case of using Decision Trees and the above specified loss, is summarized below.

< IMAGE FOR GRADIENT BOOSITING FOR REG WITH MSE>


## Gradient Boosting vs. AdaBoost (for Regression)

Another ensemble model based on boosting is [AdaBoost](). Although both models share the same idea of iteratively improving the model, there is a substantial difference on how the shortcommings of the developed model are defined. A comparison of both methods, is summarized in the following table.  

| Gradient Boosting | AdaBoost |
|:------------------|:---------|
| The model is iteratively improved using [Boosting]().| The model is iteratively improved using [Boosting]().|
| The next model is improved, by reducing the [Loss Function]() of the current weak learner.| Data samples that give wrong predictions get more weight for the next weak learner. |
| The most common weak learners used are pruned Decision Trees. | The most common weak learners used are pruned Decision Trees. |
| Every weak learner gets the same weight for the final prediction. | Every weak learner gets different weight (influence) to the final prediction, which depends on how well it performs.|

## Advantages & Disadvantages Gradient Boosted Trees

**Pros**

* Can deal with missing data and outlier
* Can deal with umerical and categorical data
* flexible, any loss function can be used

**Cons**

## Gradient Boosting in Python

What are default values in sklearn? max_nr_leaves, n_estimators, learning_rate

## Summary

## Further Reading

[1] Friedman 1999
[2] Wikipedia

fast & accurate

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

