%matplotlib inline

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import mglearn

# 1. load the dataset

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# breast cancer dataset
# {
#   target_names: ['malignant', 'benign'],
#   feature_names: [
#     'mean radius',
#     'mean texture',
#     'mean perimeter',
#     'mean area',
#     ...
#   ],
#   data: [
#     [1.79900000e+01, 1.03800000e+01, ...],
#     ...
#   ],
#   target: [0, 1, ...]
# }

# 2. See benign vs. malignant frequencies in the data

frequencies = zip(cancer.target_names, np.bincount(cancer.target))

print("Sample counts per class:\n{}".format(frequencies))
