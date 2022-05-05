import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm



data = pd.read_csv(r"Advertising.csv")
data



data.columns
plt.figure(figsize=(16, 8))
plt.scatter(
    data['TV'],
    data['sales']
)
plt.xlabel("TV ")
plt.ylabel("Sales ")
plt.show()

X = data['TV'].values.reshape(-1,1)
y = data['sales'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
reg = LinearRegression()
reg.fit(X_train, y_train)




print(reg.coef_[0][0])
print(reg.intercept_[0])

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))


# In[24]:



predictions = reg.predict(X_test)

plt.figure(figsize=(16, 8))
plt.scatter(
    data['TV'],
    data['sales']
)
plt.plot(
    X_test,
    predictions,
    linewidth=2,
    color='red'
)
plt.xlabel("TV ")
plt.ylabel("Sales ")
plt.show()

X=X_train
y=y_train
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[14]:



print('Train Score :', reg.score(X_train,y_train))
print('Test Score:', reg.score(X_test,y_test))


# In[15]:


from sklearn import metrics
print('MSE :', metrics.mean_squared_error(y_test,predictions))
