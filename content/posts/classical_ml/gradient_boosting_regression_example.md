+++
title = 'Gradient Boost for Regression - Example'
date = 2024-04-09T22:55:13-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Regression"]
categories = ["Data Science", "Machine Learning", "Gradient Boosting", "Ensemble", "Boosting", "Tree Methods", "Regression"]
keywords = ["Data Science", "Machine Learning", "Gradiend Boosting", "Ensemble", "Tree Methods", "Regression"]
images = ['/images/']
+++

## Introduction

In this post we will go through the development of a Gradient Boosting model for a regression problem considering a simplified example. We will calculate the individual steps defined and explained in the seperate post [Gradient Boost for Regression - Explained]({{< ref "/posts/classical_ml/gradient_boosting_regression.md">}}). Please refer to this post for a more general and detailed explanation of the algorithm.

## Data

We will use a simplified dataset consisting of only 10 samples, which describes how many meters a person has climbed, depending on their age, whether or not they like height, and whether or not they like goats. We used that same data in previous posts, such as [Decision Trees for Regression - Example]({{< ref "/posts/classical_ml/decision_tree_regression_example.md">}}), and [Adaboost for Regression - Example]({{< ref "/posts/classical_ml/adaboost_example_reg.md">}}) 

!["data"](/images/gradient_boosting/data.png)
*The data used in this post.*

## Build the Model

We build a Gradient Boost model with pruned Decision Trees as week learners using the above dataset. For that we follow the steps summarized in the following plot. For a more detailed explanation please refer to [Gradient Boost for Regression - Explained]({{< ref "/posts/classical_ml/gradient_boosting_regression.md">}}).

![Gradient Boosting for Regression](/images/gradient_boosting/gradient_boosting_algorithm_reg.png)
*Gradient Boosting Algorithm simplified for a regression task.*

**Step 1 - Initialize the model with a constant value -** $F_0(x) = \bar{y}$. 

The initialitation of the model, is done by taking the means of all target valuesi $(y)$. In our case

$$F_0(x) = \frac{1}{10}(200 + 700 + 600 + 300 + 200 + 700 + 300 + 700 + 600 + 700) = 500.$$

To evaluate how the model evolves, we calculate the [mean squared error (MSE)]({{< ref "/posts/ml_concepts/loss_functions.md#loss_reg">}}) after each iteration.

$$MSE(y, F_0(x)) = \frac{1}{10}((200 - 500)^2 + (700 - 500)^2 + (600 - 500)^2 + $$
$$(300 - 500)^2 + (200 - 500)^2 + (700 - 500)^2 + (300 - 500)^2 + $$
$$(700 - 500)^2 + (600 - 500)^2 + (700 - 500)^2) = 44 000$$

**Step 2 - For $m=1$ to $M=2$:**

The second step is a loop, which sequentially updates the model by fitting a weak learner, in our case a pruned Decision Tree to the residual of the target values and the previous predictions. The number of loops are the number of weak learners considered. Because the data considered in this post is so simple, we will only loop twice, i.e. $M=2$. 

**First loop $M=1$**

**2A. Compute the (pseudo-)residuals of the preditions and the true observations.**

We compute the residual as a vector

$$r_1 = y - F_0(x) = ((200 - 500), (700 - 500), (600 - 500), (300 - 500), $$
$$(200 - 500), (700 - 500), (300 - 500), (700 - 500), (600 - 500), (700 - 500))$$

$$r_1 = (-300, 200, 100, -200, -300, 200, -200, 200, 100, 200)$$

**2B. anc 2C. Fit a model (weak learner) to the residuals and find the optimized solution.**

Now we fit a Decision Tree to the residuals with the original target values. In this example we set *max_depth=3* in the Decision Tree to prune it.
We will not develop the Decision Tree in detail, but will use the result from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html). To follow a step by step example of building a Decision Tree for Regression, please refer to the seprate article [Decision Trees for Regression - Example]({{< ref "/posts/classical_ml/decision_tree_regression_example.md" >}}). 

["first tree"](/images/gradient_boosting/gb_first_tree.png)
*First Decision Tree, i.e. first weak learner*

**2D. Update the model.**

$$F_1(x) = F_0(x) + pred = ((500 - 300), (500 + 200), (500 + 100), (500 + -200), (500 -250), (500 + 200), (500 - 250), (500 - 200), (500 + 100), (500 + 200))$$

$$F_1(x) = (200, 700, 600, 300, 250, 700, 250, 700, 600, 700)$$

The MSE of these new predictions are

$$MSE(y, F_1(x)) = \frac{1}{10}((200 - 200)^2 + (700 - 700)^2 + (600 - 600)^2 + $$
$$(300 - 300)^2 + (200 - 250)^2 + (700 - 700)^2 + (300 - 250)^2 + $$
$$(700 - 700)^2 + (600 - 600)^2 + (700 - 700)^2) = 500$$

We can see that the error after this first update is much lower.

**Second loop $M=2$**

**2A. Compute the (pseudo-)residuals of the preditions and the true observations.**

$$r_2 = y - F_1(x) = ((200 - 200), (700 - 700), (600 - 600), (300 - 300), $$
$$(200 - 250), (700 - 700), (300 - 250), (700 - 700), (600 - 600), (700 - 700))$$

$$r_2 = (0, 0, 0, 0, -50, 0, -50, 0, 0, 0)$$

**2B. and 2C. Fit a model (weak learner) to the residuals and find the optimized solution.**

["second tree"](/images/gradient_boosting/gb_second_tree.png)
*Second Decision Tree, i.e. second weak learner*

**2D. Update the model.**

$$F_2(x) = F_1(x) + pred = ((200 + 0), (700 + 0), (600 + 0), (300 + 0), (250 - 50), (700 + 0), (250 + 50), (700 + 0), (600 + 0), (700 + 0))$$

$$F_2(x) = (200, 700, 600, 300, 200, 700, 300, 700, 600, 700)$$

The MSE of this updated prediction is

$$MSE(y, F_2(x)) = \frac{1}{10}((200 - 200)^2 + (700 - 700)^2 + (600 - 600)^2 + $$
$$(300 - 300)^2 + (200 - 200)^2 + (700 - 700)^2 + (300 - 300)^2 + $$
$$(700 - 700)^2 + (600 - 600)^2 + (700 - 700)^2) = 0$$

That is we see another reduction in the error. Because of the simplicity of the data, in this case the error is already $0$ after two iterations. In reality with more complex data the number of weak learners is much higher. The default value in [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) is $100$.

**Step 3 - Output final model $F_M(x)$.**

$$F_2(x) = (200, 700, 600, 300, 200, 700, 300, 700, 600, 700)$$

## Fit a Model in Python

Python's [sklearn]() library provides a [gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) package. We will use this library to fit a simple model to our example data. You can find a more complex example with a more realistic dataset on [kaggle](https://www.kaggle.com/pumalin/gradient-boosting-tutorial).

## Summary

---
If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}

