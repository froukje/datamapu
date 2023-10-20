+++
title = 'Supervised versus Unsupervised Learning - Explained'
date = 2023-10-17T09:46:31+02:00
featured_image = '/images/20231017_supervised_unsupervised/supervised_unsupervised.gif'
tags = [ "Data Science"]
categories = [ "Data Science" ]
draft = true
+++

## Machine Learning

In classical programming the programmer defines specific rules which the program follows and these rules lead to an output. In contrast Machine Learning uses data to find the rules that describe the relationship between input and output. This process of finding the rules is called 'learning'. Supervised and Unsupervised Learning are two different types of Machine Learning. Let's discover what each means.

![supervised_unsupervised](/images/20231017_supervised_unsupervised/supervised_unsupervised.gif)
*Fig. 1: Supervised and Unsupervised Learning are different types of Machine Learning.*
## Supervised Learning

Supervised learning is a technique that needs labeled data. The labels are the output that we give the model during training. The model learns a relationship between the input data and their labels. After the model is trained, this learned relationship is later used to make predictions for input data, where the labels are unknown. Supervised learning can be divided in [regression](https://en.wikipedia.org/wiki/Regression_analysis) and [classification](https://en.wikipedia.org/wiki/Statistical_classification) problems. In a classification problem we try to classify as many inputs as possible to the correct class. In a regression problem, we try to get as close as possible to the true labels with our predicted labels. A difficulty in supervised learning is, that getting the labels for a dataset can be time consuming and expensive.

![supervised](/images/20231017_supervised_unsupervised/supervised.gif)
*Fig. 2: Illustration of supervised learning.*
### Examples

1. **Classification.**
An example for supervised learning is the classification of images. For example - as illustrated in Fig. 2 - if we have images of different types of animals and we want to classify the images by the type of animal, that is shown. In this case the labels are the type of animal for each image. After the model is trained we can use it to make predictions of the type of animal on a picture.

2. **Regression.** 
An example for a regression problem is the prediction of house prices, depending on different features like size, number of rooms and location. In this case we need a dataset that contains all the information of features and the corresponding prices - which are the labels - in order to train a model. Once the model is trained and has learned the relationship between the input features and the price, we can use it to predict the price of a house, for which all the input features are known. 

## Unsupervised Learning

In contrast to supervised learning, unsupervised learning does not need labels. 

Unsupervised learning is a branch of machine learning that focuses on discovering patterns and relationships within data that lacks pre-existing labels or annotations. Unlike supervised learning, unsupervised learning algorithms do not rely on labeled examples to learn from. Instead, they aim to discover inherent structures or clusters within the data.

![unsupervised](/images/20231017_supervised_unsupervised/unsupervised.gif)
*Fig. 3: Illustration of Unsupervised learning.*

### Examples

1. **Clustering.** The data is categorized in different groups - called clusters by identifying patterns in the data. The number of desired clusters can be customized and the data within each cluster is related with each other. An example is the clustering of customers of a company based on different features such as age, gender, and income. This can be used to target the different groups (clusters) with individual marketing campains. 

2. **Dimensionality Reduction.** Dimensionality reduction can e.g. be used to reduce the feature space and to remove unimportant features. 

3. **Association**

## Summary

the strengths of each approach lie in different applications. Supervised machine learning will learn the relationship between input and output through labelled training data, so is used to classify new data using these learned patterns or in predicting outputs. 

Unsupervised machine learning on the other hand is useful in finding underlying patterns and relationships within unlabelled, raw data. This makes it particularly useful for exploratory data analysis, segmenting or clustering of datasets, or projects to understand how data features connect to other features for automated recommendation systems.
