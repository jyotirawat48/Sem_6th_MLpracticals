# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_breast_cancer


# In[2]:


df = load_breast_cancer()
#df


# In[3]:


from sklearn.model_selection import train_test_split


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(df.data,df.target,test_size=0.2, stratify=df.target,random_state=42)


# In[5]:


X_test.shape


# In[6]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=100)


# In[7]:


model.fit(X_train, y_train)


# In[8]:


model.coef_


# In[9]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1)
model.fit(X_train, y_train)
model.coef_


# In[10]:


### Coefficent's Value (decrease) if C value (decrease)
### Coefficent's Value (increase) if C value (increase)


# In[11]:


y_predicted = model.predict(X_test)


# In[ ]:
