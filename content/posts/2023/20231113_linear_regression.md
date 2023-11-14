+++
title = 'Linear Regression - Explained'
date = 2023-11-13T21:20:41+01:00
draft = true
featured_image = '/images/20231113_regression/regression.jpg'
tags = ["Data Science", "Machine Learning", "Regression"]
categories = ["Data Science", "Machine Learning", "Regression"]
keywords = ["Data Science", "Machine Learning", "Deep Learning", "Regression"]
+++

## Introduction 

Linear Regression is a type of [Supervised Machine Learning]({{< ref "20231017_supervised_unsupervised#supervised" >}} "Supervised Machine Learning") Algorithm, where a linear relationship between the input feature(s) and the target value is assumed. Linear Regression is a specific type of regression model, where the mapping learned by the model describes a linear function. As in all regression problem, the target variable is continuous. In a linear regression the linear relationship between one ([Simple Linear Regression]({{< ref "#slr" >}})) or more ([Multiple Linear Regression]({{< ref "#mrl" >}})) independent variable and one dependent variable is modelled.

![regression example](/images/20231001_regression_metrics/regression_example.jpg)
*Illustration of a simple linear regression between the body mass and the maximal running speed of an animal.*

## Simple Linear Regression{#slr}

A Simple Linear Regression describes a relationship between one independent variable ($x$) and one dependent variable ($y$). This relationship is modelled by a linear equation. The objective is to find the linear line that fits the data best, in the sense of minimizing the [Mean Squared Error]({{< ref "20231001_regression_metrics" >}}) between the predicted values and the actual values. We will discuss later, how exactly this is achieved. 

$$\hat{y} = a\cdot x +b$$
(image!)

### Asumptions



**Linear Relationship**

**Normal Distribution of Residuals**

**Multicollinearity**

**Autocorrelation**

**Homoscedasticity**

### Gradient Descent for Linear Regression(?)
To fine a linear model we need to estimate a slop and an interception.


### Interpretation of Regression Coefficients

The slope represents the change in the dependent variable for each unit change in the independent variable, while the intercept represents the predicted value of the dependent variable when the independent variable is zero.

## Multiple Linear Regression{#mlr}

Multiple linear regression establishes the relationship between independent variables (two or more) and the corresponding dependent variable. Here, the independent variables can be either continuous or categorical.

### Multicollinearity

### Advantages

**Easy Implementation**

**Interpretability**

**Scalability**

### Evaluation

After fitting a model, we need to evaluate it. To evaluate a linear regression the same metrics as for all regression problems can be used. Two very common ones are *Root Mean Squared Error (RMSE)* and *Mean Absolute Error (MAE)*. Both metrics are based on the difference between the predicted and the actual values, the so-called *Residuals*. The MAE is defined as the sum of the absolute values of the residuals for all data points, divided by the total number of data points. The RMSE is defined as the squre root of the sum of the squared residuals divided by the total number of data points. Both metrics avoid the elimination of errors by taking the absolute value and the square accordingly and are easy to interpret, because they carry the same units as the data. The RMSE, due to taking the square, helps to reduce large errors. A more detailed overview and description about these and other common [metrics for regression]({{< ref "20231001_regression_metrics" >}} "Metrics for Regression") is given in a separate article.


### Extrapolation
Whenever a linear regression model is fit to a group of data, the range of the data should be carefully observed. Attempting to use a regression equation to predict values outside of this range is often inappropriate, and may yield incredible answers. This practice is known as extrapolation. Consider, for example, a linear model which relates weight gain to age for young children. Applying such a model to adults, or even teenagers, would be absurd, since the relationship between age and weight gain is not consistent for all age groups.

## Linear Regression in Python
