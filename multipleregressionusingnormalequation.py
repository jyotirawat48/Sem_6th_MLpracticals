import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
col_list = ["TV", "radio","newspaper","sales"]
data = pd.read_csv("Advertising.csv", usecols=col_list)
data.head()
x1 = data["TV"]
x2 = data["radio"]
x3 = data["newspaper"]
y = data["sales"]
print(x1.shape)
print(x2.shape)
print(x3.shape)

#plot Tv vs Sale
plt.scatter(x1,y)
plt.xlabel("TV")
plt.ylabel("Sale")
#plot Radio vs Sale
plt.scatter(x2,y)
plt.xlabel("radio")
plt.ylabel("Sale")
#plot Radio vs Sale
plt.scatter(x3,y)
plt.xlabel("newspaper")
plt.ylabel("Sale")
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
y = np.array(y)
n = len(x1)
n

x_bias = np.ones((n,1))
x1_new = np.reshape(x1,(n,1))
x2_new = np.reshape(x2,(n,1))
x3_new = np.reshape(x3,(n,1))
x_new = np.append(x_bias,x1_new,axis=1)
x_new = np.append(x_new,x2_new,axis=1)
x_new = np.append(x_new,x3_new,axis=1)
X_new

x_new_transpose = np.transpose(x_new)
x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)
temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)
temp_2 = x_new_transpose.dot(y)
theta = temp_1.dot(temp_2)
theta

beta_0 = theta[0]
beta_1 = theta[1]
beta_2 = theta[2]
beta_3 = theta[3]

print(beta_0)
print(beta_1)
print(beta_2)
print(beta_3)

def predict_values(beta_0,beta_1,beta_2,beta_3,tv,radio,newspaper):
    predicted_value = beta_0 + tv*beta_1 + radio*beta_2 + newspaper* beta_3
    return predicted_value

tv = 10
radio = 20
newspaper = 30
print(predict_values(beta_0,beta_1,beta_2,beta_3,tv,radio,newspaper)
