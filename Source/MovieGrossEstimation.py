#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the dataset

dataset = pd.read_csv("Data/movie_metadata.csv")

X = dataset.loc[:, dataset.dtypes == (np.float64 or np.int64)]
X = X.loc[:, X.columns != 'gross']
y = dataset.loc[:, dataset.columns == 'gross']

#Data Cleaning

X = X.fillna(X.mean())
y = y.fillna(y.mean())

X = X.astype(int)
y = y.astype(int)

np.any(np.isnan(y))

np.all(np.isfinite(X))

X = X.astype(np.float64, copy = False)
y = y.astype(np.float64, copy = False)

# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying Naive Bayes Classification Algorithm

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train.values.ravel())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.values.ravel(), y_pred)



