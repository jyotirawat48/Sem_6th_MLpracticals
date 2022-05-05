#suplots
plt.figure(figsize=(8,8))
plt.subplot(221)
plt.plot(x,x**2, label="2018")
plt.plot(x,x**3, label="2017")

plt.subplot(222)
plt.hist(vals,50)

plt.subplot(223)
plt.pie(values,labels=labels,radius=1)

plt.subplot(224)
plt.pie(values,labels=labels,radius=1,explode=(0,0,1,1,0,1),shadow=True)
plt.show()
