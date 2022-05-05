# In[45]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer


# In[46]:


df = load_breast_cancer()
#df


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


#X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.9)
X_train, X_test, Y_train, Y_test = train_test_split(df.data,df.target,test_size=0.2, stratify=df.target,random_state=42)


# In[49]:


X_test.shape


# In[50]:


Y_test.shape


# In[51]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=0.1)


# In[53]:


model.fit(X_train, Y_train)


# In[54]:


#X_test


# In[41]:


y_predicted = model.predict(X_test)


# In[42]:



print('Accuracy on the training subset: {:.3f}'.format(model.score(X_train, Y_train)))
print('Accuracy on the test subset: {:.3f}'.format(model.score(X_test, Y_test)))


# In[43]:


#X_test


# In[44]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_predicted)


# In[57]:


from sklearn.metrics import plot_confusion_matrix


# In[ ]:
