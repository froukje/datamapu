+++
title = 'Gradient Descent'
date = 2024-02-27T20:55:28-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Deep Learning", "Neural Nets"]
categories = ["Data Science", "Machine Learning", "Deep Learning", "Neural Nets"]
keywords = ["Data Science", "Machine Learning", "Deep Learning", "Neural Nets"]
images = ['/images/gradient_descent/gradient_descent.png']
+++

## Introduction

Gradient Descent is a mathematical optimization technique, which is used to find local minima of a function. In Machine Learning it is used in a variety of models such as [Gradient Boosting]({{< ref "gradient_boosting_regression">}} "gradient_boosting") or [Neural Networks]({{< ref "/posts/deep_learning/intro_dl.md">}}) to minimize the [Loss Function]({{< ref "/posts/ml_concepts/loss_functions.md">}}). It is an iterative algorithm that takes small steps towards the minimum in every iteration. The idea is to start at a random point and then take a small into the direction of the steepest descent of this point. Then take another small step into the direction of the steepest descent of the next point and so on. The direction of the steepest descent is determined using the gradient of the function, so its name *Gradient Descent*.

![gradient_descent](/images/gradient_descent/gradient_descent.png)
*Gradient Descent illustrated.*

## Intuition

Gradient Descent is often compared to descending from the top of a mountain to a valley. To reach the valley, we take individual steps. After each step, we check which is the direction of the steepest descent for the next step and we move into this direction. The stepsize we take may vary and effects how long we need to reach the minimum and if we reach it at all. Very small steps means that it will take very long to get to the valley. However, if the steps are too large, especially close to the minimum, we may miss it by taking a step over it. 

![gradient_descent](/images/gradient_descent/mountain1_small.jpg)
*Gradient Descent intuition.*


## Definition

Gradient Descent is an iterative method that aims to find the minimum of a differentiable function. In the context of Machine Learning this is usually the loss function $L = L(w_i)$, which depends on the model parameters. To find a local minimum, we start at an arbirtrary random point and move into the direction of the steepest descent at this point. Mathematically, the direction of the steepest ascent at a specific point is defined by the gradient at this point, which is also called the slope. Consequently, the direction of the steepest descent at a point is the negative of the gradient at this point. At this next point we again move a small step into the direction of the negative of the gradient. In this way we iteratievly approach the minimum. This procedure can be formulated as

$$w_{i+1} = w_{i} - \alpha \cdot \nabla_{w_i} L(w_i), $$ 

with $\alpha$ the step size we take into the direction of the negative of the gradient. $\alpha$ is a hyperparameter, which is called the *learning rate*. How big the learning rate is influences the convergance of the algorithm. If the learning rate is very low, a lot of iterations are needed to get to the minimum. On the other hand, if the learning rate is too large, we may overpass the minimum. One possibility is to start with a larger learning rate and make it smaller over time. Gradient Descent does not garanty to reach a local or even global minimum. It is only certain to approach a stationary point that satisfies $\nabla_{w_i} L(w_i) = 0$.

![gradient_descent](/images/gradient_descent/learning_rate.png)
*Small and large learning rate illustrated.*

### Vanishing Gradients

## Variants of Gradient Descent

### Stochastic Gradient Descent

### Batch Gradient Descent

## Summary

---
If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}
