+++
title = 'Logistic Regression - Explained'
date = 2023-12-02T09:31:24+01:00
draft = true
featured_image = ''
tags = ["Data Science", "Machine Learning", "CLassification", "Logistic Regression"]
categories = ["Data Science", "Machine Learning", "Classification"]
keywords = ["Data Science", "Machine Learning", "Deep Learning", "Classification", "Logistic Regression"]
+++

## Introduction

Logistic Regression is a [Supervised Machine Learning]({{< ref "20231017_supervised_unsupervised#supervised" >}} "Supervised Machine Learning") algorithm, where a model is developed, that relates the target variable to one or more input variables (features). However, in contrast to [Linear Regression]({{< ref "20231113_linear_regression.md" >}} "Linear Regression") the target (dependent) variable is not numerical, but [categorical](https://en.wikipedia.org/wiki/Categorical_variable). That is the target variable can be classified in different categories (e.g.: 'test passed' or 'test not passed'). An idealized example of two categories for the target variable is illustrated in the plot below. The relation described in this example is whether a test is passed or not, depending on the amount of hours studies. Note, that in real world examples the border between the two classes depending on the input feature (independ variable) will usually not be as clear as in this plot.

![logistic regression](/images/20231202_logistic_regression/logistic_regression_simplified.png)
*Simplified and idealized example of a logistic regression*

## Binary Logistic Regression

If the target variable contains two classes we speak of a Binary Logistic Regression. The target values for binary classification are usually denominated as 0 and 1. In the previous plot 'test passed' is the class 1 and 'test not passed' is class 0. To develop a Logistic Regression model we start with the equation of a Linear Regression. We start with assuming that we only have one input feature (independend variable), that is we use the equation for a [Simple Linear Regression]({{< ref "20231113_linear_regression.md#slr" >}} "Linear Regression").

$$ \hat{y} = a\cdot x + b, $$

with $\hat{y}$ simulating the target values $y$ and $x$ being the input feature. Using this Linear Regression model, would give us values for $\hat{y}$ ranging between $-\infty$ and $+\infty$. However, we finally want only $0$ and $1$ as outputs. To achieve that we define a model that does not predict the outcome ($0$ or $1$) directly, but the probability of the event. Then this probability is mapped to the values $0$ and $1$ by defining a threshold. For probabilites below this threshold the predicted target is $0$ and for probabilities above it is $1$. Usually this threshold is $0.5$, but it can be customized. To predict a probabilty the output values need to be between $0$ and $1$. A function that satisfies this is the Sigmoid function, which is defined as

$$f(z) = \frac{1}{1 + e^{-z}}.$$

![sigmoid function](/images/20231202_logistic_regression/sigmoid.png)
*The sigmoid Function.*

We then take our Linear Regression equation and plug it into the logistic equation. With that we get

$$f(x) = \hat{y} = \frac{1}{1 + e^{-(a\cdot x + b)}},$$

a Sigmoid Function depending on the parameters $a$ and $b$. How the Sigmoid Function changes depending on these parameters is illustrated below.

![logistic regression](/images/20231202_logistic_regression/logistic_function.png)
*The function for Logistic Regression illustrated for different parameters.*

This is our  final Logistic Regression model. 

![logistic regression](/images/20231202_logistic_regression/linear_logistic.png)
*From Linear Regression to Logistic Regression*

If more than one independent variable (input feature) is considered, the input features can be numerical or categorical as in a Linear Regression. The exact same idea as described above is followed, but using the equation for [Multiple Linear Regression]({{< ref "20231113_linear_regression.md#mlr" >}} "Linear Regression") as input for the Sigmoid function

$$\hat{y} = \frac{1}{1 + e^{-(a_0 + a_1\cdot x_1 + a_2\cdot x_2 + \cdots + a_n\cdot x_n)}}.$$

## Multinomial Logistic Regression

If the target variable can take more than two classes, we speak of Multinomial Logistic Regression. By definition a Logistic Regression is designed for binary target variables. If we consider more than two target variables, we need to adapt the algorithm. This can be done by splitting the multiple classification problem into several binary classification problems. The two most common ways to do that are
the *One-vs-Rest* and *One-vs-One* strategies. As example consider the example of predicting one of the animals dog, cat, rabbit, or ferrit by specific criteria, that is we have four classes in our target data.

**One-vs-Rest.** This approach creates $N$ binary classification problems, if N is the number of classes in tha target data. Then each class is predicted against all the others. In the above example four binary problems would be considered to solve the multiple classification problem.

* Classifier 1: class 1: dog, class 2: cat, rabbit, or ferrit
* Classifier 2: class 1: cat, class 2: dog, rabbit, or ferrit
* Classifier 3: class 1: rabbit, class 2: dog, cat, or ferrit
* Classifier 4: class 1: ferrit, class 2: dog, cat, or rabbit


With these classifiers we get probabilites for each category (dog, cat, rabbit, ferrit) and the maximum of these probabilites is taken for the final outcome. As example consider the following outcome:

* Classifier 1: probability for dog (class 1): 0.5, probability for cat, rabbit, or ferrit (class 2): 0.5
* Classifier 2: probability for cat (class 1): 0.7, probability for cat, rabbit, or ferrit (class 2): 0.3
* Classifier 3: probability for rabbit (class 1): 0.2, probability for cat, rabbit, or ferrit  (class 2): 0.8
* Classifier 4: probability for ferrit (class 1): 0.3, probability for dog, cat, or rabbit (class 2): 0.7

In this case the highest probability achieves classifier 2, that is the final outcome would be a probability of 0.7 for the class 'dog'.


**One-vs-One.** In this approach a binary model is fitted for each binary combination of the output classes. That is for $N$ classes $N \choose 2$ models are created. For the example with four classes this results in

* Classifier 1: class 1: dog, class 2: cat
* Classifier 2: class 1: dog, class 2: rabbit
* Classifier 3: class 1: dog, class 2: ferrit
* Classifier 4: class 1: cat, class 2: rabbit
* Classifier 5: class 1: cat, class 2: ferrit
* Classifier 6: class 1: rabbit, class 2: ferrit

To define the prediction of the multiclass classification problem from these classifiers, the number of predictions for each class is counted and the class with highest number of correct predictions among classifiers is the final prediction. This procedure is also known as "Voting".

Note that a disadvantage of these methods is that they require to fit multiple models, which takes long if the dataset considered is large. For the  *One-vs-One* even more than for the *One-vs-Rest* method.

## Find Best Fit

As in all supervised Machine Learning models we estimate the model parameters, in this case $a_0$, $a_1$, $\dots$, $a_n$ as parameters to optimize the model. This is done by minimizing a loss function, which describes the error between the actual values and the predictions with respect to these parameters. In the case of a Linear Regression problem this loss function is the mean squared error (also known as least squares optimization). For a Logistic Regression, we need to define a different loss function. A common choice in this case is the *[Negative Log Likelihood Function ($NLL$)]()*, which can be derived from the likelihood function. This is however a topic on its own. You can find a separate article with a detailed explanation [here](). 

## Interpretation

Let's consider the defined model function

$$\hat{y} = \frac{1}{1 + e^{-(a_0 + a_1\cdot x_1 + a_2\cdot x_2 + \cdots + a_n\cdot x_n)}}.$$

With reformulation, we get

...

we can define the odds
 
... calculate odds ...

These describe the chances $\frac{p}{1-p}$. These chances are a measure for the separation of the two classes. That is although Linear Regression and Logistic Regression are used for different types of problems (regression vs. classification), they still have a lot in common. In both cases a line (one input feature) or a hyperplane (more than one input feature) is used. In a Linear Regression however this line / hyperplane is used to predict the target variable, while in Logistic Regression it is used to separate two classes. In Logistic Regression the interpreation of the coefficients $a_0$, $a_1$, $\dots$, $a_n$ however, is not as straight forward as in the case of the Linear Regression, because the relationship is not linear any more. Only the sign of the coefficients tells us if the probability is increased or decreased.

## Evaluation

## Application

## Example

```Python
# some code
echo "Hello world"
```

Simple Python Example

Kaggle Notebook
