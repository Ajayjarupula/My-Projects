#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries

# In[55]:


import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

from warnings import filterwarnings
filterwarnings(action='ignore')


# # Importing Iris Dataset

# In[56]:


dataset=pd.read_csv('C:/Users/admin/Downloads/iris/iris.data',names=['sepalLength','sepalwidth','petalLength','petalwidth','class'])


# In[57]:


dataset


# In[58]:


print(dataset.shape)


# In[59]:


print(dataset.describe())


# In[60]:


dataset.head(100)


# In[61]:


dataset.tail(100)


# In[62]:


dataset.dtypes


# In[63]:


x = dataset.iloc[:, [0,1,2,3]].values
x


# In[64]:


y = dataset.iloc[:, 4].values
y


# In[65]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0) 


# In[66]:


# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = GaussianNB()
classifier.fit(x_train, y_train) 


# In[67]:


y_pred = classifier.predict(x_test)

print("Predicted values:")
print(y_pred)


# # Perform the Accuracy of Iris dataset

# In[83]:


#Using Navie Bayes Classification
acc= accuracy_score(y_test,y_pred)*100
print ("\n\nAccuracy of Navie Bayes: ", acc)


# In[84]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[85]:


ig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1,2), ticklabels=('Predicted Setosa', 'Predicted Versicolor', 'Predicted Virginica'))
ax.yaxis.set(ticks=(0,1,2), ticklabels=('Actual Setosa', 'Actual Versicolor', 'Actual Virginica'))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


# In[86]:


dataset.isnull()


# In[87]:


print(dataset['class'].describe())


# In[88]:


print(dataset['class'].value_counts())


# In[89]:


dataset.plot(kind='box',subplots=True,layout=(7,2),figsize=(15,20))


# In[90]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.axis('equal')
l=['versicolor','setosa','virginica']
s=[50,50,50]
ax.pie(s, labels=l,autopct='%1.2f%%')
plt.show()


# In[91]:


dataset.hist()
plt.show()


# In[92]:


dataset.plot(kind='density',subplots=True,layout=(2,2),sharex=False)


# In[93]:


sns.pairplot(dataset,hue='class');

