+++
title = 'Ensemble Models - Illustrated'
date = 2023-12-26T11:24:29+01:00
draft = true
featured_image = ''
tags = ["Data Science", "Machine Learning", "Regression", "Classification"]
categories = ["Data Science", "Machine Learning", "Classification", "Regression"]
keywords = ["Data Science", "Machine Learning", "Classification", "Regression"]
+++

## Introduction

In *Ensemble Learning* multiple Machine Learning models are combined to one single prediction in order to improve the predictive skill. The individual models can be of different types or the same. Ensemble learning is based on ["the wisdom of the crowds"](https://en.wikipedia.org/wiki/The_Wisdom_of_Crowds), which assumes that the expected value of multiple estimates is more accurate than the estimate of a single model. Three main types of Ensemble Learning methods are most common.

## Bagging / Bootstrap Aggregation

In *Bagging* $N$ random samples are drawn from the trainingsdata with replacement, this is called bootstrapping. Next, $N$ models are trained for the task of interest, one for each of one drawn dataset. These models are independent of each other and can be trained in parallel. If a classification problem is considered, the final prediction is the majority vote of all predictions, that is the class mostly predicted by the single models. if a regression problem is considered, the final prediction is the mean of all predictions.

Bagging is of often used to reduce the variance of a single model. The most famous example of bagging is the [Random Forest](), which uses a set of [Decision Trees]() to make a combined prediction. 

<IMAGE>


## Boosting

In Boosting the individual models used are trained sequentially and not in parallel. Each newly trained model builds on the previous one.

<IMAGE>

## Stacking

In stacking a set of different types of models are used, which output are used as an input to a meta model, which provides the final prediction.

<IMAGE>
