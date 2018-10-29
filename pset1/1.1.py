
# coding: utf-8

# # Logistic regression
# 
# - Newton's model

# In[15]:


import numpy as np

X = np.loadtxt(path('data/logistic_x.txt'))
y = np.loadtxt(path('data/logistic_y.txt'))

theta = np.zeros(X.shape[1])

print(X[:5])
print(y[:5])
print(theta)

