#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
print('Python: {}'.format(sys.version))
import scipy
print("Scipy: {}".format(scipy.__version__))
import numpy
print("Numpy: {}".format(numpy.__version__))
import pandas
print("Pandas: {}".format(pandas.__version__))
import sklearn
print("Sklearn: {}".format(sklearn.__version__))
import matplotlib
print("Matplotlib: {}".format(matplotlib.__version__))


# In[14]:


import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier


# In[15]:


url="https://raw.githubusercontent.com/Amit366/data/master/iris.csv"
dataset=pandas.read_csv(url)


# In[16]:


#dimensions
print(dataset.shape)


# In[17]:


#take a peek into the data
dataset.head(20)


# In[18]:


#sanatical summary
print(dataset.describe())


# In[19]:


#class distribution
print(dataset.groupby('variety').size())


# In[20]:


#univariate plots - box and whisker type
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
pyplot.show()


# In[21]:


#histogram of the variables
dataset.hist()
pyplot.show()


# In[22]:


#multivariant plot
scatter_matrix(dataset)
pyplot.show()


# In[23]:


#creating validation dataset
#splitting dataset
array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train,x_validation,y_train,y_validation=train_test_split(x,y,test_size=0.2,random_state=1)


# In[31]:


#logistic Regression
#Linear Discriminant Analysis
#K-Nearest Neighbor
#Classification and Regression Trees
#Gaussian Naive Bayes
#Support Vector Machine

# Creating models
models=[]
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


# In[32]:


#Evaluating the models
results=[]
names=[]
for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=1)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s %f (%f)'% (name,cv_results.mean(),cv_results.std()))


# In[33]:


#compare our models
pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparision')
pyplot.show()


# In[38]:


#make a prediction
model=SVC(gamma='auto')
model.fit(x_train,y_train)
prediction=model.predict(x_validation)


# In[39]:


#evaluate our predictions
print(accuracy_score(y_validation,prediction))
print(confusion_matrix(y_validation,prediction))
print(classification_report(y_validation,prediction))

