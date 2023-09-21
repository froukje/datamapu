+++
title = 'Metrics in Classification Problems — An Overview'
date = 2023-09-20T00:08:49+02:00
draft = true
featured_image = '/images/20230920_classification_metrics/classification_metrics.gif'
tags = [ "Data Science", "Classification", "Metrics"]
categories = [ "Data Science" ]
+++

## Metrics in Classification Problems - An Overview

![classification metrics](/images/20230920_classification_metrics/classification_metrics.gif)

### Classification Problems

Supervised Machine Learning projects can be divided into [regression](https://en.wikipedia.org/wiki/Regression_analysis) and [classification](https://en.wikipedia.org/wiki/Statistical_classification) problems. In regression problems, we predict a continuous variable (e.g. temperature), while in classification, we classify the data into discrete classes (e.g. classify cat and dog images). A subset of classification problems is the so-called [binary classification](https://en.wikipedia.org/wiki/Binary_classification), where only two classes are considered. An example of this would be classifying e-mails as spam and no-spam or cat images versus dog images. When we consider more than two classes, we speak of [multi-class classification](https://en.wikipedia.org/wiki/Multiclass_classification). An example of this is the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset, where images of handwritten digits are classified, i.e. we have 10 different classes. Depending on the problem we consider, we need different metrics to evaluate our Machine Learning model. Within each of these two types of Machine Learning problems, we have to choose which of the metrics fits best our needs. This post is supposed to be short and focused on giving an overview of the most common metrics for **binary classification** problems, what they mean, and when to use them.

### Confusion Matrix
A good way to start evaluating a classification problem is the so-called *confusion matrix*. This table gives an overview of how many predictions were correct and how many were not for each class. Let's imagine we want to classify cat and dog images and our model gives the following result on 10 test images:

![classification table](/images/20230920_classification_metrics/classification1.gif "Example true and predicted values.")

From this table, we can see, when our model was correct and when not. The confusion matrix is a way to illustrate these results more clearly. We can use the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) package to display the confusion matrix in Python. One important step before applying it is to replace the string values with integers in the table above. In this case, we categorize dog images as 1 and cat images as 0.

![classification table](/images/20230920_classification_metrics/classification2.jpg "Example true and predicted values categorized as 0 and 1.")

We can then display the confusion matrix as follows:

```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["cat", "dog"])
disp.plot();
```

![confusion matrix](/images/20230920_classification_metrics/classification3.png "Confusion matrix.")

The confusion matrix shows us, the predicted labels versus the true labels, and the numbers in the matrix give the number of images that were predicted for each case. The first column, first row of the matrix (cat - cat) shows how many results predicted a cat and the real image was also a cat, The first column, second row (cat-dog) shows how many results predicted a cat, but the true label was a dog and so on. Note, that this not only divides into correct and incorrect, but also in the type of error we make. There are four possible outcomes:

* *TP (True Positive)*: The prediction is a dog (positive) and the true image is also a dog (positive)
* *FN (False Negative)*: The prediction is a cat (negative), but the true image is a dog (positive)
* *FP (False Positive)*: The prediction is a dog (positive), but the true image is a cat (negative)
* *TN (True Negative)*: The prediction is a cat (negative) and the true image is a cat (negative).

### Metrics
Now, finally the metrics. With the above concepts of TP, FN, FP, and TN we can define several metrics that evaluate binary classification problems in different aspects. In the table below you can find the most common metrics for binary classification problems, how to calculate them and when to use them.

![confusion matrix](/images/20230920_classification_metrics/classification4.png "Most common metrics for binary classification.")

Note: These metrics can be extended for multi-class classification. An overview can be found in this [blog](https://medium.com/r/?url=https%3A%2F%2Fwww.analyticsvidhya.com%2Fblog%2F2021%2F07%2Fmetrics-to-evaluate-your-classification-model-to-take-the-right-decisions%2F).

### Further Reading

* https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc
* https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/
