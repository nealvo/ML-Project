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

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('iris.csv')

# Display basic information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())
```

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis delves deeper into relationships between features, distributions, and potential patterns. Visualizations such as pairplots, boxplots, and correlation heatmaps provide a comprehensive view.
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Pairplot to visualize relationships between features
sns.pairplot(df, hue='target_names', palette='viridis')
plt.suptitle('Pairplot of Iris Dataset', y=1.02)
plt.show()

# Boxplot for each feature by target class
plt.figure(figsize=(15, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x='target_names', y=feature, data=df, palette='viridis')
    plt.title(f'Boxplot of {feature} by Target Class')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
```
## Machine Learning Tasks
### K-Nearest Neighbors (KNN) Classifier
The KNN classifier is applied to predict the species of iris flowers based on their measurements. Model accuracy and visualizations such as confusion matrices and decision regions are included.
```python
from sklearn.neighbors import KNeighborsClassifier

# Extract features for training
X = df.iloc[:, :-2]
y = df['target']

# Initialize the K-Nearest Neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
knn_classifier.fit(X, y)

# Make predictions on the testing data
y_pred = knn_classifier.predict(X)
```
### Clustering with K-Means
K-Means clustering groups similar instances together based on their features. The clusters are visualized to explore potential patterns and relationships within the dataset.
```python
from sklearn.cluster import KMeans

# Extract features for clustering
X_clustering = df.iloc[:, :-2]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_clustering)
```

##Conclusion

This repository provides a comprehensive analysis of the Iris dataset, encompassing data analysis, exploratory data analysis, and machine learning tasks. The visualizations and insights generated aim to enhance understanding and facilitate further exploration.

Feel free to clone this repository, experiment with the code, and explore additional machine learning tasks on the Iris dataset.
