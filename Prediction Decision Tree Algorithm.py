#!/usr/bin/env python
# coding: utf-8

# # Prediction Using Decision Tree Algorithm:

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import seaborn as sns


# In[2]:


dataset=pd.read_csv("C:/Users/admin/Downloads/Logistic_Iris.csv")


# In[3]:


dataset.head(10)


# In[5]:


x = dataset.iloc[:, [0,1,2,3]].values
x


# In[6]:


y = dataset.iloc[:, 4].values
y


# In[7]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)


# In[8]:


sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)


# In[9]:


#Tree induction using Gini Index
dtree_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=3,min_samples_leaf=5)
dtree_gini.fit(xtrain, ytrain)


# In[10]:


y_pred1 = dtree_gini.predict(xtest)
print("Predicted values:")
y_pred1


# In[11]:


accgini= accuracy_score(ytest,y_pred1)*100
print ("\n\nAccuracy using Gini Index: ", accgini)


# In[12]:


cm = confusion_matrix(ytest, y_pred1)
print ("\n\n Confusion Matrix -using Gini Index: \n", cm)


# In[18]:


fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0,1,2), ticklabels=('Predicted Setosa', 'Predicted Versicolor', 'Predicted Virginica'))
ax.yaxis.set(ticks=(0,1,2), ticklabels=('Actual Setosa', 'Actual Versicolor', 'Actual Virginica'))
ax.set_ylim(2.5, -0.5)
for i in range(3):
    for j in range(3):
         ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()


# In[19]:


print("\n\nClassification Report – Using Gini Index: \n",classification_report(ytest, y_pred1))


# # Plotting Decision Tree Using Gini Index

# In[20]:


tree.plot_tree(dtree_gini)

