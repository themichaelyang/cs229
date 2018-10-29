
# coding: utf-8

# # Logistic regression
# 
# - Newton's model

# In[233]:


import numpy as np

raw_X = np.loadtxt(path('data/logistic_x.txt')) # m x n
raw_y = np.loadtxt(path('data/logistic_y.txt')) # 1 x m

y = np.array([1 if v == 1 else 0 for v in raw_y])

x_0 = np.ones(X.shape[0]).reshape(-1, 1) # (m,) => (m, 1)
X = np.concatenate((x_0, raw_X), axis=1) # m, n+1

theta = np.zeros(X.shape[1])

print(X[:5])
print(y)
print(theta)


# ## Newton's method
# 
# Update rule: $\theta := \theta - H^{-1} \nabla_{\theta} l(\theta)$
# 
# Partial derivative vector and Hessian:
# $$
# \begin{aligned}
# \nabla_{\theta} \ell (\theta)_{j} &= \sum_{i = 1}^m (y^{(i)} - h_{\theta}(x^{(i)}))x_j^{(i)} \\
# \\
# H_{kj} &= \frac{\partial^2 \ell(\theta)}{\partial \theta_k \partial \theta_j} \\
# &= \sum_{i = 1}^m x_j^{(i)} x_k^{(i)} g(\theta^T x^{(i)}) (1 - g(\theta^T x^{(i)}))
# \end{aligned}
# $$

# In[247]:


def sigmoid(z):
    '''vectorized sigmoid'''
    return 1 / (1 + np.exp(-z))

def hypothesis(theta, X):
    '''vectorized hypothesis, X = design matrix'''
    return sigmoid(X @ theta) # result is length m vector

def partials(theta, X, y):
    '''vectorized partial derivative'''
    h = hypothesis(theta, X)
    residuals = (y - h)
    # jth index of this vector = sum over all training: res * jth feature
    return residuals @ X # result is length n vector

def hessian(theta, X, y):
    prod = hypothesis(theta, X) * (1 - hypothesis(theta, X)) # m-vector
    D = np.diag(prod) # construct diagonal matrix of sigmoid products
    return -X.T @ D @ X # error: was missing the - sign!! spent so long on this

# for y = {-1, 1}
# def cost(theta, X, y):
#     m = X.shape[0]
#     return np.sum(np.log(1 + np.exp( (-y - (X @ theta) ))) / m

# def log_likelihood(theta, X, y):
    


# In[248]:


print(partials(theta, X, y))
print(hessian(theta, X, y))
print(hypothesis(theta, X))


# In[249]:


def newton(theta, X, y, threshold = 0, max_iter = 15):

    delta = 1000
#     cost_history = []
    iterations = 0
    
    while delta >= threshold and iterations <= max_iter:
        args = (theta, X, y)
        theta -= np.linalg.pinv(hessian(*args)) @ partials(*args)
#         c = cost(*args)
#         cost_history.append(c)
        delta = 10
        iterations += 1
        print(theta)
        
    return theta


# In[250]:


theta = np.zeros(X.shape[1])
theta = newton(theta, X, y)
print(theta)

