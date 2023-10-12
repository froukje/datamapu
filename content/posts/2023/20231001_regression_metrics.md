+++
title = 'Metrics for Regression Problems'
date = 2023-09-30T21:24:12+02:00
draft = true
featured_image = '/images/20231001_regression_metrics/regression_metrics.jpg'
tags = [ "Data Science", "Regression", "Metrics"]
categories = [ "Data Science" ]

+++

## Metrics for Regression Problems  
### An Overview

![regression metrics](/images/20231001_regression_metrics/regression_metrics.jpg)

### Regression Problems

[Regression](https://en.wikipedia.org/wiki/Regression_analysis) problems in Machine Learning are a type of supervised learning problems, where a continuous numerical variable is predicted, such as for example the age of a person or the price of a product. A special type is the [linear regression](https://en.wikipedia.org/wiki/Linear_regression), where a linear relationship between two ([simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)) or more (multiple linear regression) is analized. The example plots in this article will be illustrated with a simple linear regression, however the metrics introduced here are common metrics for all types of regression problems, including multiple linear regression and non-linear regression. The simple linear regression is only chosen or illustration purpose. 
 

![regression example](/images/20231001_regression_metrics/regression_example.jpg)
*Illustration of a simple linear regression between the body mass and the maximal running speed of an animal.*

### Residuals

In regression problems the predicted results are rarely exactly the same as the true values, but lie either a bit above or below them. The difference between true and predicted values are a measure of goodness for the prediction and are defined as residuals. Metrics for regression problems are usually based on residuals. 
![residuals](/images/20231001_regression_metrics/residuals.jpg)
### Metrics

With the just defined concept of residuals, we can define different metrics, that are useful for different error measurements.

![mae](/images/20231001_regression_metrics/mae.jpg)
![mse](/images/20231001_regression_metrics/mse.jpg)
![rmse](/images/20231001_regression_metrics/rmse.jpg)
![mape](/images/20231001_regression_metrics/mape.jpg)
![r_squared](/images/20231001_regression_metrics/r_squared.jpg)
![adjusted_r_squared](/images/20231001_regression_metrics/adj_r_squared.jpg)

### Example

https://en.wikipedia.org/wiki/Linear_regression

### Summary
