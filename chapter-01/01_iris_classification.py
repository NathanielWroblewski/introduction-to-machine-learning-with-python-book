%matplotlib inline

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import mglearn

# 1. load the dataset

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# iris dataset
# {
#   target_names: ['setosa', 'versicolor', 'virginica'],
#   feature_names: [
#     'sepal length (cm)',
#     'sepal width (cm)',
#     'petal length (cm)',
#     'petal width (cm)'
#   ],
#   data: [
#     [5.1, 3.5, 1.4, 0.2], ...
#   ],
#   target: [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]
# }

# 2. shuffle and separate test set from training set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
  iris_dataset['data'], iris_dataset['target'], random_state=0)

# X_train.shape (112, 4) 75% data
# X_test.shape (38, 4)   25% data
# y_train.shape (112,)   75% target
# y_train.shape (38,)    25% target

# 3. Look  at the data.  Making a pair-plot by converting numpy
# arrays into pandas data frames (data, labels)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=mglearn.cm3)

# 4. Run k-nearest neighbors

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# 5. Make a prediction

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# 6. Determine accuracy

y_pred = knn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Test set score: {:.2f}".format(accuracy))
