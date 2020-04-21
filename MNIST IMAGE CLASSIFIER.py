#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as ply
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


d=pd.read_csv('mnist_test.csv')
d.head()


# In[11]:


#extracting the data
data=d.iloc[2,1:].values


# In[12]:


#reshaping
data=data.reshape(28,28).astype('uint8')
data


# In[13]:


ply.imshow(data)


# In[17]:


#separating label and pixels
df_x=d.iloc[:,1:]
df_y=d.iloc[:,0]


# In[44]:


#train and test
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.4,random_state=4)


# In[45]:


#calling rf classifier
rf=RandomForestClassifier(n_estimators=100)


# In[46]:


#fitting model
rf.fit(x_train,y_train)


# In[47]:


#prediction
pred=rf.predict(x_test)


# In[48]:


#count correct predictions
s=y_test.values
c=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        c+=1
c


# In[49]:


len(pred)


# In[50]:


#accuracy
c/len(pred)

