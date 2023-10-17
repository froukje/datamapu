+++
title = 'Metrics for Regression Problems'
date = 2023-09-30T21:24:12+02:00
draft = false
featured_image = '/images/20231001_regression_metrics/regression_metrics.jpg'
tags = [ "Data Science", "Regression", "Metrics"]
categories = [ "Data Science" ]

+++

## Metrics for Regression Problems  
### An Overview

![regression metrics](/images/20231001_regression_metrics/regression_metrics.jpg)

### Regression Problems

[Regression](https://en.wikipedia.org/wiki/Regression_analysis) problems in Machine Learning are a type of supervised learning problem, where a continuous numerical variable is predicted, such as, for example, the age of a person or the price of a product. A special type is the [linear regression](https://en.wikipedia.org/wiki/Linear_regression), where a linear relationship between two ([simple linear regression](https://en.wikipedia.org/wiki/Simple_linear_regression)) or more (multiple linear regression) is analyzed. The example plots in this article will be illustrated with a simple linear regression. However the metrics introduced here are common metrics for all types of regression problems, including multiple linear regression and non-linear regression. The simple linear regression is only chosen for illustration purposes. 
 

![regression example](/images/20231001_regression_metrics/regression_example.jpg)
*Illustration of a simple linear regression between the body mass and the maximal running speed of an animal.*

### Residuals

In regression problems, the predicted results are rarely exactly the same as the true values, but lie either a bit above or below them. The difference between true and predicted values are a measure of goodness for the prediction and are defined as residuals. Metrics for regression problems are usually based on residuals. 
![residuals](/images/20231001_regression_metrics/residuals.jpg)
### Metrics

With the just defined concept of residuals, we can define different metrics that are useful for different error measurements.

![mae](/images/20231001_regression_metrics/mae.jpg)
![mse](/images/20231001_regression_metrics/mse.jpg)
![rmse](/images/20231001_regression_metrics/rmse.jpg)
![mape](/images/20231001_regression_metrics/mape.jpg)
![r_squared](/images/20231001_regression_metrics/r_squared.jpg)
![adjusted_r_squared](/images/20231001_regression_metrics/adj_r_squared.jpg)

### Example

Let's consider the above example, relating the body mass of an animal with the maximal running speed, and calculate the RMSE. In order to do that, we first need to calculate the predictions. We use the *LinearRegression* method from [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to fit a linear regression and print the predictions.


```
from sklearn.linear_model import LinearRegression

d = {'animal': ['horse', 'black rhino', 'giraffe', 'pronghorn', 'cheetah', 'wildebeest'], 
     'body_mass': [400, 1400, 1000, 50, 60, 300], 'max_speed': [70, 45, 60, 100, 110,  90]}
df = pd.DataFrame(data=d)

x = df['body_mass'].values.reshape(-1,1) 
y_true = df['max_speed'].values.reshape(-1,1)
reg = LinearRegression()
reg.fit(x, y_true)
y_pred = reg.predict(x)
```

This results is the following predictions (rounded to three decimals).

```
y_pred = [100.047 ,  99.617,  89.284,  84.979, 59.147,  41.926]
```

Using the formula from the previous section to calculate the RMSE, we get (rounded to three decimal).

![rmse_by_hand](/images/20231001_regression_metrics/rmse_by_hand.png)

In Python we can define our custom function to calculate the rmse.

```
import numpy as np

def rmse(y_true, y_pred):
   return np.sqrt(np.sum((y_true - y_pred)**2)/(y_true.shape[0]))

```

Alternatively, we can also use [sklearn](https://scikit-learn.org/stable/modules/model_evaluation.html) to calulate the RMSE. 

```
from sklearn.metrics import mean_squared_error

mean_squared_error(y_true, y_pred, squared=False)
```

Both giving the same result 7.559 km/h as we calculated by hand.

Note, if *squared=True* in the *mean_squared_error* method, the MSE is calculated instead of the RMSE.

### Summary

In this article we learned about the most often used metrics to measure the performance of regression problems and when to use them. They can be used for linear and non-linear regression and are generally based of the *Residual Error* between the true and the predicted value.
