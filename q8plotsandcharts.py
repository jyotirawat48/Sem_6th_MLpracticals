import matplotlib.pyplot as plt
import numpy as np
x=np.arange(40)
x
plt.plot(x)
plt.show()
y=np.sin(x)
plt.plot(x)
plt.plot(y)
plt.show()

y=np.cos(x)
plt.plot(x,color='green')
plt.plot(y)
plt.show()

x=np.array([1,2,3,4,5])
y=x**2
plt.scatter(x,y,color="orange", label="squares of number", marker="o") #color, label, marker are optional arguments
plt.legend()
plt.xlabel("numbers")
plt.ylabel("squares")
plt.show()

x=np.array([1,2,3,4,5])
plt.plot(x,x**2,label="squares of number", marker="o")
plt.plot(x,x**3,label="cubes of number", marker="o")
plt.legend()
plt.xlabel("numbers")
plt.title("Squares and Cubes")
plt.show()

numbers=np.random.randint(0,10,5)
indices=np.arange(5)
indices=indices+2015
numbers2=np.random.randint(0,10,5)
indices2=np.arange(5)
indices2=indices2+2015
print(numbers)
print(numbers2)
plt.bar(indices,numbers,0.25,color="red",label="rainfall")
plt.bar(indices2+0.25,numbers2,0.25,color="green",label="humidity")
plt.legend()
plt.show()

labels=["english","hindi","maths","science","social science","computers"]
values=[90,80,40,73,78,43]
plt.pie(values,labels=labels,radius=1)
plt.show()
plt.pie(values,labels=labels,radius=1,explode=(0,0,1,0,0,0),shadow=True)
plt.show()


u=5
sigma=2
vals=u+sigma*np.random.randn(1000)
print(vals.shape)
plt.hist(vals,50)
plt.show()
