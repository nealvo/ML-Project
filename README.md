# Iris Dataset Analysis and Machine Learning

![Iris Flowers](images/iris_flowers.jpg)

## Overview

This repository showcases a comprehensive analysis of the famous Iris dataset. The dataset includes measurements of sepal length, sepal width, petal length, and petal width for three species of iris flowers: setosa, versicolor, and virginica.

## Table of Contents

- [Data Analysis](#data-analysis)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning Tasks](#machine-learning-tasks)
  - [K-Nearest Neighbors (KNN) Classifier](#k-nearest-neighbors-knn-classifier)
  - [Clustering with K-Means](#clustering-with-k-means)
- [Conclusion](#conclusion)

## Data Analysis

The initial data analysis provides insights into the structure of the dataset. Basic information, summary statistics, and visualizations help understand the characteristics of the features and target labels.


## Data Analysis

The initial data analysis provides insights into the structure of the dataset. Basic information, summary statistics, and visualizations help understand the characteristics of the features and target labels.

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('iris.csv')

# Display basic information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())
