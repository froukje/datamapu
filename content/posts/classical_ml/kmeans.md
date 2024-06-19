+++
title = 'Understanding K-Means Clustering'
date = 2024-06-13T21:10:46-03:00
draft = true
tags = ["Data Science", "Machine Learning", "Clustering", "unsupervised"]
categories = ["Data Science", "Machine Learning", "Clustering", "unsupervised"]
keywords = ["Data Science", "Machine Learning", "Clustering", "unsupervised"]
images = ['/images/kmeans/3_clusters.png']
+++

## Introduction

*K-Means* is an example for a *clustering* algorithm. Clustering is a fundamental concept in Machine Learning, where the goal is to group a set of objects in such a way that objects in the same group are more similar to each other than to those in other groups. Clustering belongs to the set of [unsupervised]({{< ref "/posts/ml_concepts/supervised_unsupervised#unsupervised">}}) Machine Learning algorithms, that is no ground truth is needed. Among the various clustering algorithms, *K-Means* stands out for its simplicity and efficiency. In this blog post, we will explain the algorithm behind K-Means, and see how to implement it in Python.

{{< img-slider-kmeans >}}


*Illustration of clustering.*

## What is K-Means Clustering? - The Algorithm

K-Means clustering is an unsupervised Machine Learning algorithm used to partition an unlabeled, unsorganised data set into k clusters, where the number k is pre-defined in advance. Each data point is assigned to the cluster with the closest centroid of this cluster, based on a predefined distance metric. The algorithm minimizes the inner-cluster variance. More precisely the algorithm has the following steps

**1. Initialization:** The algorithm starts by selecting k initial centroids. These can be chosen randomly or by using more sophisticated methods like [K-Means++](https://en.wikipedia.org/wiki/K-means%2B%2B) to improve convergence. When using the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans) library the initial method "k-means++" is the default method.

**2. Assignment Step:** For each data point the distance to each of the centroids is calculated. Each data point is then assigned to the nearest centroid. As distance metric typically the Euclidean distance is chosen. After this step k clusters are formed.

**3. Update Step:** The centroids are recalculated as the mean of all data points assigned to each cluster. 

**Repeat steps 2 and 3** iteratively until the centroids no longer change significantly or the indicated number of iterations is reached. The k clusters after the convergence criterion is reached is the final result. 

< illustration of the algorithm >

## Choosing the Right Number of Clusters

One of the key challenges in K-Means clustering is selecting the optimal number of clusters. A number of methods have been developed to determine the optimal number of clusters. In this post, we will only discuss the most popular ones, which are the *Elbow Method* and the *Silhouette Score*. 

### Elbow Method

The *Elbow Method* helps in determining the optimal number of clusters by plotting the sum of squared distances (inertia) from each point to its assigned centroid against the number of clusters. As the number of clusters increases, the inertia decreases. The point where the decrease sharply slows down (forming an "elbow") indicates the optimal number of clusters.

< image of ellbow method >

Its popularity stems from its simplicity and visual appeal. It allows data scientists and analysts to quickly identify a suitable number of clusters by visual inspection.

**Pros:**

* Simple to understand and implement.
* Provides a visual indication of the optimal number of clusters.

**Cons:**

* The elbow point can sometimes be ambiguous or not very pronounced, making it hard to decide the optimal number of clusters.


### Silhouette Method

The *Silhouette Score* or *Silhouette Coefficient* evaluates the consistency within clusters and separation between clusters. It ranges from -1 to 1, with higher values indicating better clustering. The method is in detail described on [wikipedia](https://en.wikipedia.org/wiki/Silhouette_(clustering)#:~:text=The%20silhouette%20score%20is%20specialized,distance%20or%20the%20Manhattan%20distance.). It is calculated as follows:


**1. step:** For each data point $i$ in cluster $C_I$: 

**Calculate the mean inner-cluster distance (a):** This is the average distance between the data point $i$ and all other points in the same cluster.

$$a(i) = \frac{1}{|C_I| - 1} \sum_{j\in C_I, j\neq i} d(i,j),$$

with $|C_I|$ the total number of data points in cluster $I$, and $d(i, j)$ the distance between data point $i$ and data point $j$.

**Calculate the mean nearest-cluster distance (b):** This is the average distance between the data point $i$ and all points in the nearest neighboring cluster.

$$b(i) = \min_{J\neq I}\frac{1}{C_J}\sum_{j\in C_J} d(i,j),$$

where $C_J$ is any other cluster than $C_I$. Since the minimum over all clusters is taken, this results in the average distance of data point $i$ and all data points in the neighboring cluster. The neighboring cluster is the next best fit cluster for point i.

**2. step: Calculate the silhouette coefficient for the data point $i$:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}.$$

This equation holds if $|C_I| > 1$, if $|C_I| = 1$, $s(i)$ is defined as $0$.

The above equation can be rewritten as 

$$
s(i) = \begin{cases} 
1 - \frac{a(i)}{b(i)} & \text{if } a(i) < b(i) \newline
0 & \text{if } a(i) = b(i) \newline
\frac{b(i)}{a(i)} - 1 & \text{if } a(i) > b(i)
\end{cases}
$$

s(i) ranges from -1 to 1:
* s(i) close to 1: The data point is well-clustered. The mean distance to the data points within the cluster is smaller than the mean distance to the data points of the neighboring cluster.
* s(i) close to 0: The data point is on or very close to the decision boundary between two neighboring clusters.
* s(i) close to -1: The data point might have been assigned to the wrong cluster. The mean distance to the data points in neighboring cluster is smaller than the mean distance to the data points within the cluster.

**3. step: Calculate the mean silhouette score for all data points:** The overall silhouette score for the clustering is the average silhouette coefficient for all data points.

$$S = \frac{1}{N}\sum_{i=1}^N s(i)$$

where $N$ is the total number of data points.

< image of silhoutte method >

This method is favored because it provides a more detailed and rigorous evaluation of clustering quality, helping to ensure that the chosen number of clusters truly represents the data structure.

**Pros:**

* Provides a clear and quantitative measure of clustering quality.
* Works well for a variety of cluster shapes and densities.

**Cons:**

* Computationally more intensive than the Elbow Method, especially for large datasets.
* May not be as intuitive to interpret as the Elbow Method.

**Other Methods**

The Elbow method and the Silhouette Score are the most popular and the most straight forward methods. There are however several other methods, that are more complex. We are not going to explain them here, but only name a few and link to a reference.

* Gap Statistic
* Davies-Bouldin Index
* Information Criterion Approaches (BIC/AIC) 

## Applications

This algorithm is widely used in fields like market segmentation, image compression, and pattern recognition.

## Code Example

´´´
class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init='auto', max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='lloyd')[source]
´´´ 

elbow method 

Silhouette Score

## Summary

---
If this blog is useful for you, please consider supporting.

{{< bmc-button slug="pumaline" >}}

