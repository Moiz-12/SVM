#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv("iris.csv")
print (df)


# In[5]:


df.head()


# In[6]:


df.replace('setosa',1,inplace=True)
df.replace('versicolor',2,inplace=True)
df.replace('virginica',3,inplace=True)


# In[7]:


df['species'].unique()


# In[8]:


X=df.drop('species',axis=1)


# In[41]:


y=df[['species']]


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=424)


# In[84]:


#SVM


# In[85]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# In[86]:


clf_svm=SVC()


# In[87]:


clf_svm.fit(X_train,y_train)


# In[88]:


output=clf_svm.predict(X_test)


# In[89]:


output


# In[90]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,output)*100


# In[91]:


from sklearn.model_selection import KFold


# In[92]:


SVC1=SVC()
param_grid={'C':[0.2,0.5,1,5,10,20],'gamma':['scale','auto'],'kernel':['linear','rbf']}
randomized_search=RandomizedSearchCV(SVC1,param_grid)


# In[93]:


randomized_search.fit(X_train,y_train)


# In[94]:


randomized_search.best_params_


# In[95]:


clf_tuned=SVC(C=1,kernel='rbf',gamma='scale')


# In[96]:


clf_tuned.fit(X_train,y_train)


# In[97]:


output=clf_tuned.predict(X_test)


# In[98]:


output


# In[99]:


accuracy_score(y_test,output)*100


# In[100]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,output)


# In[103]:


from sklearn.metrics import classification_report


# In[104]:


y_true,y_prod=y_test,clf_svm.predict(X_test)
print('Reprot')
print(classification_report(y_true,y_prod))


# In[ ]:




