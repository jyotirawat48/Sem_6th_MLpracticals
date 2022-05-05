#Q. Implement Simple Linear Regression using analytical method and depict model on scatter data plot.
#a) Take x=[1,2,4], y=[2,3,6]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x=np.array([1,2,4])
y=np.array([2,3,6])
x_mean=np.mean(x)
print(x_mean)
y_mean=np.mean(y)
print(y_mean)
n=x.size
print(n)

ss_xy=np.sum(x*y) - n*x_mean*y_mean
print(ss_xy)
ss_xx=np.sum(x*x) - n*x_mean*x_mean
print(ss_xx)
b1=ss_xy/ss_xx
print(b1)
b0=y_mean-b1*x_mean
print(b0)

plt.scatter(x,y)
y_pred=b0+b1*x
print(y_pred)
plt.plot(x,y_pred,color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression[x~y]")
plt.show()

#b) Regress Sales~Radio from Advertisingdata.csv
col_list = ["TV", "radio","newspaper","sales"]
data = pd.read_csv("Advertising.csv", usecols=col_list)

data["sales"]     #x-axis
data["radio"]     #y-axis
x_mean=np.mean(data["sales"])
print(x_mean)
y_mean=np.mean(data["radio"])
print(y_mean)
n=data["sales"].size
print(n)
ss_xy=np.sum(data["sales"]*data["radio"]) - n*x_mean*y_mean
print(ss_xy)
ss_xx=np.sum(data["sales"]*data["sales"]) - n*x_mean*x_mean
print(ss_xx)
b1=ss_xy/ss_xx
print(b1)
b0=y_mean-b1*x_mean
print(b0)

plt.scatter(data["sales"],data["radio"])
y_pred=b0+b1*data["sales"]
print(y_pred)
plt.plot(data["sales"],y_pred,color='red')
plt.xlabel("sales")
plt.ylabel("radio")
plt.title("Linear Regression[Sales~Radio]")
plt.show()

