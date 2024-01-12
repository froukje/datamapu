+++
title = 'Supervised versus Unsupervised Learning - Explained'
date = 2023-10-17T09:46:31+02:00
tags = [ "Data Science", "Machine Learning"]
categories = [ "Data Science" , "Machine Learning"]
draft = false
keywords = ["Data Science", "Machine Learning", "Supervised Learning", "Unsupervised Learning"]
images = ['/images/20231017_supervised_unsupervised/supervised_unsupervised.gif']
+++

## Machine Learning

In classical programming, the programmer defines specific rules which the program follows and these rules lead to an output. In contrast, Machine Learning uses data to find the rules that describe the relationship between input and output. This process of finding the rules is called 'learning'. Supervised and Unsupervised Learning are two different types of Machine Learning. Let's discover what each means.

![supervised_unsupervised](/images/20231017_supervised_unsupervised/supervised_unsupervised.gif)
*Fig. 1: Supervised and Unsupervised Learning are different types of Machine Learning.*
## Supervised Learning{#supervised}

Supervised learning is a technique that needs labeled data. The labels are the output that we give the model during training. The model learns a relationship between the input data and their labels. After the model is trained, this learned relationship is used to make predictions for input data, where the labels are unknown. Supervised learning can be divided into [regression](https://en.wikipedia.org/wiki/Regression_analysis) and [classification](https://en.wikipedia.org/wiki/Statistical_classification) problems. In a classification problem, we try to classify as many inputs as possible to the correct class. In a regression problem, we try to get as close as possible to the true labels with our predicted labels. A difficulty in supervised learning is that getting the labels for a dataset can be time-consuming and expensive.

![supervised](/images/20231017_supervised_unsupervised/supervised.gif)
*Fig. 2: Illustration of supervised learning.*
### Examples

1. **Classification.**
An example of supervised learning is the classification of images. For example - as illustrated in Fig. 2 - if we have images of different types of animals and we want to classify the images by the type of animal, that is shown. In this case, the labels are the type of animal for each image. After the model is trained, we can use it to make predictions of the type of animal in a picture.

2. **Regression.** 
An example of a regression problem is the prediction of house prices, depending on different features like size, number of rooms and location. In this case, we need a dataset that contains all the information about the features and the corresponding prices - which are the labels - in order to train a model. Once the model is trained and has learned the relationship between the input features and the price, we can use it to predict the price of a house, for which all the input features are known. 

## Unsupervised Learning

In contrast to supervised learning, unsupervised learning does not need labels. With unsupervised learning, the data is divided into groups by finding relationships, patterns, or similarities between the data in each group. As there are no labels, this is done by only analyzing the data. 

![unsupervised](/images/20231017_supervised_unsupervised/unsupervised.gif)
*Fig. 3: Illustration of Unsupervised learning.*

### Examples

1. **Clustering.** The data is categorized into different groups - called clusters - by identifying patterns in the data. The number of desired clusters can be customized and the data within each cluster is related with each other. An example is the clustering of customers of a company based on different features such as age, gender, and income. This can be used to target the different groups (clusters) with individual marketing campaigns. 

2. **Dimensionality Reduction.** Dimensionality reduction can e.g. be used to reduce the feature space and to remove unimportant features. The objective is to decrease the complexity of a dataset while keeping the most important information. It however also decreases the possibility to interprete the results.

3. **Association** Association rules are used to find relationships within a dataset. A common use case is to find products that relate to the ones in your basket on online platforms.

## Summary

Supervised and unsupervised learning are two different types of Machine Learning with different objectives. In supervised learning, we try to map the input data to the labels. That is, we try to find a map that describes this relationship. A trained model can be used to make predictions for new data, when the labels are unknown. Unsupervised learning is not used to make predictions, but to find underlying patterns in the data. It can e.g. be used for data exploration or feature space reduction.

If this blog is useful for you, I'm thankful for your support!
{{< bmc-button slug="pumaline" >}}


