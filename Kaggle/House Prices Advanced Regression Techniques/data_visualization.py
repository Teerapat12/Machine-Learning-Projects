#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


cols = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target =['SalePrice']
X_train = df_train[cols]
X_test = df_test[cols]
y_train = df_train[target]


# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X_train)
y = sc_y.fit_transform(y_train)

# Fitting the SVR to the dataset  (Support Vector Regression)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

X_test = X_test.fillna(X_test.median())

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))

