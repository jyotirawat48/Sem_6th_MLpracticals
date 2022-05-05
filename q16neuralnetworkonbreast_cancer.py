# In[1]:


from numpy import*
import pandas as pd 
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# In[2]:


df = load_breast_cancer()


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(df.data,df.target,test_size=0.2,random_state=4)
y_test


# In[4]:


nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(10,15),random_state=1)


# In[5]:


nn.fit(X_train,y_train)


# In[6]:


pred=nn.predict(X_test)
pred


# In[7]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,pred) 


# In[ ]:


