#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
t_df=pd.read_excel(r"C:\Users\742711\Documents\ML\titanic.xls")
print(len(t_df))


# In[2]:


sns.countplot(x="survived" , data=t_df)


# In[3]:


sns.countplot(x='survived',hue='sex',data=t_df)


# In[4]:


sns.countplot(x='survived',hue='pclass',data=t_df)


# In[5]:


t_df['age'].plot.hist()


# In[8]:


t_df['fare'].plot.hist()


# In[11]:


sns.countplot(x='sibsp',hue='survived',data=t_df)


# In[12]:


t_df.isnull()


# In[13]:


t_df.isnull().sum()


# In[17]:


sns.heatmap(t_df.isnull(), yticklabels=False,cmap='viridis')


# In[20]:


sns.boxplot(x='pclass',y='age',data=t_df)


# In[21]:


t_df.head(5)


# In[25]:


t_df.drop('boat',axis=1,inplace=True)


# In[26]:


t_df.drop('body',axis=1,inplace=True)


# In[27]:


t_df.head(10)


# In[28]:


t_df.dropna(inplace=True)


# In[29]:


t_df.head(10)


# In[30]:


print(len(t_df))


# In[32]:


sns.heatmap(t_df.isnull(), yticklabels=False,cmap='viridis')


# In[33]:


t_df.isnull().sum()


# In[61]:


sex=pd.get_dummies(t_df['sex'],drop_first=True)
sex.head(5)


# In[66]:


embaked=pd.get_dummies(t_df['embarked'],drop_first=True)
embaked.head(5)


# In[63]:


p_class=pd.get_dummies(t_df['pclass'],drop_first=True)
p_class.head(5)


# In[64]:


t_df=pd.concat([t_df,sex,embark,p_class],axis=1)


# In[65]:


t_df.head(10)


# In[68]:


t_df.drop(['pclass','name','sex','ticket','embarked'],axis=1,inplace=True)


# In[69]:


t_df.head(10)


# In[70]:


t_df.drop(['cabin','home.dest','C'],axis=1,inplace=True)


# In[71]:


t_df.head(10)


#  ###training and testing
#  

# <h1>sap</h1>

# In[81]:


X=t_df.drop('survived',axis=1)
y=t_df['survived']


# In[82]:


from sklearn.cross_validation import train_test_split


# In[ ]:


train_test_split


# In[83]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1)


# In[84]:


from sklearn.linear_model import LogisticRegression


# In[85]:


logmodel=LogisticRegression()


# In[86]:


logmodel.fit(X_train,y_train)


# In[87]:


prediction=logmodel.predict(X_test)


# In[88]:


from sklearn.metrics import classification_report


# In[89]:


classification_report(y_test,prediction)


# In[90]:


from sklearn.metrics import confusion_matrix


# In[91]:


confusion_matrix(y_test,prediction)


# In[92]:


y=14+2+16+40


# In[93]:


print(54/y)


# In[ ]:




