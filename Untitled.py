#!/usr/bin/env python
# coding: utf-8

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)
df= pd.read_csv(r'C:\Users\742711\Documents\ML\name.csv')
X=df['Open'].values
Y=df['Close'].values

mean_x=np.mean(X)
mean_y=np.mean(Y)

m=len(X)

num=0
den=0

for i in range(m):
    num+=(X[i]-mean_x)*(Y[i]-mean_y)
    den+=(X[i]-mean_x)**2
    
b1=num/den
b0=mean_y-b1*mean_x

print(b1,b0)


# In[48]:


max_x=np.max(X)+1
min_x=np.min(X)-1
x=np.linspace(min_x,max_x,10)
y=b0+b1*x
plt.plot(x,y,color='#58b970',label='Regression Line')
plt.plot(X,Y,c='#ef5423',label='scaterring Line')
plt.xlabel("Head")
plt.ylabel("foot")
plt.legend()
plt.show()


# In[45]:



ss_t=0
ss_r=0
for i in range(m):
    y_pre=b0+b1*X[i]
    ss_t+=(Y[i]-mean_y)**2
    ss_r+=(Y[i]-y_pre)**2

r2=1-(ss_r/ss_t)
print(r2)
    


# In[47]:


print((1-r2)*100)


# In[ ]:




