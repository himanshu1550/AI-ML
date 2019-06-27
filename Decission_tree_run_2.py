#!/usr/bin/env python
# coding: utf-8

# In[ ]:


df=pd.read_csv('C:/Users/742711/Documents/ML/Decision-Tree-from-Scratch-master/Decision-Tree-from-Scratch-master/data/Titanic.csv')


# In[ ]:


train_df, test_df=train_test_split(df,test_size_proportion=0.2)
tree=decission_tree_alogrithm(train_df)
accuracy=calculate_accuracy(test_df,tree)


# <h1>Import the module</h1>

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import random
import pprint as pprint


# #load and prepare data

# # #Import data and make the required changes in the system

# In[115]:


df=pd.read_csv(r'C:\Users\742711\Documents\ML\Decision-Tree-from-Scratch-master\Decision-Tree-from-Scratch-master\data\iris.csv')
df.drop('Id',axis=1, inplace =True)
df.rename(columns={'species':'Label'},inplace=True)
df.head(10)


# In[76]:


sns.heatmap(df.isnull(), yticklabels=False,cmap='viridis')


# In[84]:


plt.plot( 'sepal_length','petal_length', data=df, linestyle='none', marker='o')
plt.show()


# In[18]:


df.isnull().sum()


# In[19]:


df.info()


# <h1>Train_test_split_df</h1>

# In[118]:


def train_test_split(df,test_size):
    if isinstance(test_size, float):
        test_size=round(test_size*len(df))

    index=df.index.tolist()
    test_index=random.sample(population=index,k=test_size)
    
    test_df=df.loc[test_index]
    train_df=df.drop(test_index)
    return  train_df, test_df


# <h1>random selection by index id</h1>

# In[119]:


random.seed(0)
train_df, test_df= train_test_split(df, test_size=0.2)


# In[120]:


len(test_df)


# In[121]:


train_df.head(5)


# 
# # Working with numpy is faster then pandas

# In[122]:


data=train_df.values
data[:]


# In[123]:


def purity_check(data):
    label_column=data[:,-1]
    unique_label_values=np.unique(label_column)
    
    if len(unique_label_values)==1:
        return True
    else:
        return False
    


# In[124]:


purity_check(train_df[train_df.petal_width< 0.2].values)
purity_check(train_df[train_df.petal_width> 0.2].values)


# # Classify

# In[125]:


def class_data(data):
    label_column=data[:,-1]  
    unique_class, class_count=np.unique(label_column,return_counts=True)
    classification=unique_class
    return classification


# In[126]:


class_data(train_df[train_df.petal_width >0.0].values)


# 
# # potential splits

# In[127]:


def potential_split(data):
    potential_splits={}
    nrow,n_columns=data.shape
    for columns in range(n_columns-1):
        potential_splits[columns]=[]
        values=data[:,columns]
        unique_values=np.unique(values)
        unique_values.sort()

        for i in range(len(unique_values)):
            if i!=0:
                current_index=unique_values[i]
                next_index=unique_values[i-1]
                potential_split=(current_index+next_index)/2
                potential_splits[columns].append(potential_split)
    
    return potential_splits


# In[128]:


potential_splits=potential_split(train_df.values)


# In[135]:


sns.lmplot(data=train_df, x='petal_width', y='petal_length',hue='Label',fit_reg=False,size=6,aspect=2)
plt.vlines(x=potential_splits[3], ymin=1,ymax=7)


# In[140]:


sns.lmplot(data=train_df, x='petal_width', y='petal_length',hue='Label',fit_reg=False,size=6,aspect=2)
plt.hlines(y=potential_splits[2], xmin=0,xmax=2.5)


# In[ ]:




