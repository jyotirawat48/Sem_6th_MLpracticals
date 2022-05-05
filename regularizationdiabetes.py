import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt 
import math
from sklearn.model_selection import train_test_split 
import array as arr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'outcome'] # load dataset 
pima = pd.read_csv("diabetes.csv", header=None, names=col_names) 
pima.head() 
target = ['outcome']
feature_cols = ['pregnant', 'insulin', 'bmi',
                'age', 'glucose', 'bp', 'pedigree']
X = pima[feature_cols] # features
y = pima.outcome # target variable
# split data into training and validation datasets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
linear_reg  = LinearRegression()
linear_reg.fit(X_train,y_train)

y_pred = linear_reg.predict(X_test)
mse_linear = mean_squared_error(y_pred,y_test)
print(mse_linear)
print(linear_reg.coef_)

#lasso regression
lambda_values = [0.000001,0.0001,0.001,0.005,0.01,0.2,0.3,0.4,0.5]

for i  in lambda_values:
    lasso_reg = Lasso(i)
    lasso_reg.fit(X_train,y_train)
    y_pred = lasso_reg.predict(X_test)
    mse_lasso =mean_squared_error(y_pred,y_test)
    print("lasso mse with lambda={} is {}".format(i,mse_lasso))
print(lasso_reg.coef_)

#ridge regression
lambda_values = [0.00001,0.01,0.05,0.1,0.5,1,1.5,3,5,6,7,8,9,10]

for i  in lambda_values:
    ridge_reg = Ridge(i)
    ridge_reg.fit(X_train,y_train)
    y_pred = lasso_reg.predict(X_test)
    mse_ridge =mean_squared_error(y_pred,y_test)
    print("lasso mse with lambda={} is {}".format(i,mse_ridge))
print(ridge_reg.coef_)
