#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
t_df=pd.read_excel(r"C:\Users\742711\Documents\ML\Weather.xlsx")
print(len(t_df))


# In[6]:


sns.countplot(x='Temperature',hue='Hour', data=t_df)


# In[15]:


plt.plot( 'Temperature', 'Wind_Speed', data=t_df, marker='o', color='mediumvioletred')
plt.show()


# In[20]:


sns.distplot( t_df["Temperature"] , color="skyblue", label="Sepal Length")
sns.distplot( t_df["Wind_Speed"] , color="red", label="Sepal Width")
sns.plt.legend()


# In[28]:


plt.plot( 'Hour','Temperature', data=t_df, linestyle='none', marker='o')
plt.show()
plt.plot( 'Hour','Wind_Speed', data=t_df, linestyle='none', marker='o')
plt.show()
plt.plot( 'Hour','Humidity', data=t_df, linestyle='none', marker='o')
plt.show()


# In[31]:


plt.plot('Hour','Wind_Direction', data=t_df)
plt.show()


# In[35]:



# use the scatter function
plt.scatter('Wind_Direction','Hour',data=t_df, alpha=1)
plt.show()


# In[ ]:




