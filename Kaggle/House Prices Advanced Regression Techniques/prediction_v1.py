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
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)
X_test = X_test.fillna(method='pad')
X_test = sc_X.transform(X_test)



# Fitting the SVR to the dataset  (Support Vector Regression)
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)

'''
# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 300,random_state=0)
regressor.fit(X_train,y_train)
'''


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train.ravel(), cv = 10)
accuracies.mean()

y_pred = sc_y.inverse_transform(regressor.predict(X_test))

ID = df_test['Id']

columns = ['Id','SalePrice']
predict_df = pd.DataFrame(
    {'Id': ID,
     'SalePrice': y_pred
     },  columns=columns)
predict_df.to_csv('house_price_predictions_v1.csv',index=False)