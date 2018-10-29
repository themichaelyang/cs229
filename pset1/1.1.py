
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


# ## Newton's method
# 
# Update rule: $\theta := \theta - H^{-1} \nabla_{\theta} l(\theta)$
# 
# Partial derivative vector and Hessian:
# $$
# \begin{aligned}
# \nabla_{\theta} \ell (\theta)_{j} &= \sum_{i = 1}^m x_j^{(i)} (y^{(i)} - h_{\theta}(x^{(i)}))x_j^{(i)} \\
# \\
# H_{kj} &= \frac{\partial^2 \ell(\theta)}{\partial \theta_k \partial \theta_j} \\
# &= \sum_{i = 1}^m x_j^{(i)} x_k^{(i)} g(\theta^T x^{(i)}) (1 - g(\theta^T x^{(i)})
# \end{aligned}
# $$
