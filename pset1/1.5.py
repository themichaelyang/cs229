
# coding: utf-8

# In[104]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

data_file = "data/quasar_train.csv"
data_test_file = "data/quasar_test.csv"

data_train = np.genfromtxt(data_file, delimiter=",")
data_test = np.genfromtxt(data_test_file, delimiter=",")

# Remember to restart kernel periodically; global namespace can be polluted


# In[2]:


def fit_row_1(data_train):
    spectrum = data_train[0, :] # first row
    flux = data_train[1, :] # second row

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(spectrum, flux, s=5, c=spectrum, cmap='cool')

    # although this is the first training example, there are actually many x and y values
    # we can actually think of it as many training examples, modeling:
    # Ax + B = y 
    # where x is one value of lambda and y is the flux

    x = np.insert(spectrum.reshape(-1, 1), 0, values=1, axis=1)
    y = flux

    theta_pred = (np.linalg.pinv(x.T @ x) @ x.T) @ y
    print(theta_pred)

    line_x = np.asarray([spectrum[0], spectrum[-1]]).reshape(-1, 1)
    print(np.insert(line_x, 0, values=1, axis=1))
    line_y = np.insert(line_x, 0, values=1, axis=1) @ theta_pred
    
    print(line_y)
    ax.plot(line_x, line_y, c='black')
fit_row_1(data_train)


# In[3]:


def get_weights(x_eval, X, tau):
    # tau = bandwidth param
    x_full = np.full(X.shape, x_eval)[:,1:]
    weights = np.exp(-((X[:,1:] - x_full) ** 2) / (2 * (tau**2)))
    W = np.diag(weights[:,0])
    return W
    
    
def fit_pt_weighted(x_eval, X, y, tau):
    W = get_weights(x_eval, X, tau)
    theta_pred = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return theta_pred

def evaluate(theta_pred, x_eval):
    return np.asarray([1, x_eval]) @ theta_pred

def fit_row_1_weighted(data_train, taus=[5]):
    spectrum = data_train[0, :]
    flux = data_train[1, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(spectrum, flux, s=5, c=spectrum, cmap='cool')

    # x really should have been caps the first time round
    X = np.insert(spectrum.reshape(-1, 1), 0, values=1, axis=1)
    y = flux

    line_x = np.arange(spectrum[0], spectrum[-1])
    
    for t in taus:
        line_y = []
        for x_eval in line_x:
            theta_pred = fit_pt_weighted(x_eval, X, y, t)
            line_y.append(evaluate(theta_pred, x_eval))
        ax.plot(line_x, line_y, c='black', alpha=0.75)
    plt.show()

fit_row_1_weighted(data_train)


# In[4]:


# Overfit:
fit_row_1_weighted(data_train, [1])

# Underfit:
fit_row_1_weighted(data_train, [100, 1000])


# In[5]:


def smooth_data(data_train, tau=5):
    spectrum = data_train[0, :]
    X = np.insert(spectrum.reshape(-1, 1), 0, values=1, axis=1)
    
    smoothed_Y = []
    for flux in data_train[1:, :]:
        y_i = flux
        smoothed_y_i = []
#         for x_eval in spectrum:
        for x_eval in np.arange(spectrum[0], spectrum[-1]):
            theta_pred = fit_pt_weighted(x_eval, X, y_i, tau)
            smoothed_y_i.append(evaluate(theta_pred, x_eval))
        smoothed_Y.append(smoothed_y_i)

    return np.asarray(smoothed_Y), np.arange(spectrum[0], spectrum[-1])

# smoothed data: y values, smoothed
# inputs: all lambdas
smoothed_data, inputs = smooth_data(data_train)


# In[6]:


print(smoothed_data.shape)

def plot_pts(x, y):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=5, c=x, cmap='cool')
    plt.show()

plot_pts(inputs, smoothed_data[0, :])

# ith row of smoothed data is the function


# In[85]:


# def f(x_eval, i, smoothed_data, inputs):
#     j = np.where(inputs == x_eval)
#     return smoothed_data[i][j]

# where f_right is evaluated from (Lyman-alpha)
right_index = np.where(inputs == 1300)[0][0]

# smoothed_data is the function! the row is the fn outputs
def f(i, smoothed_data, x_eval=None):
    if not x_eval:
        return smoothed_data[i, :]
    else:
        j = np.where(inputs == x_eval)[0][0]
        return smoothed_data[i, j]

def bound(t):
    return max(1 - t, 0)

def distance(f1_i, f2_i, smoothed_data, right=False):
    diff = (f(f1_i, smoothed_data) - f(f2_i, smoothed_data))
    if right:
        diff = diff[right_index:]
    return diff @ diff

def neighbors(f1_i, k, smoothed_data, inputs):
    distances = np.asarray([distance(f1_i, i, smoothed_data, True)
                            for i in range(smoothed_data.shape[0])])
    indicies = np.argsort(distances)
    return indicies[0:k], indicies[-1]
    

def f_left(x_eval, f_right_i, smoothed_data, inputs):
    knn, h = neighbors(f_right_i, 3, smoothed_data, inputs)
    numer = 0
    for i in knn:
        numer += bound(distance(f_right_i, i, smoothed_data, True) / h) * f(i, smoothed_data, x_eval)
    
    deno = 0
    for i in knn:
        deno += bound(distance(f_right_i, i, smoothed_data, True) / h)

    return numer / deno

print(f_left(1201, 0, smoothed_data, inputs))
print(smoothed_data[2, np.where(inputs == 1201)[0][0]])

def plot_f_left(f_right_i, smoothed_data, inputs, data_train):
    left_index = np.where(inputs == 1200)[0][0] - 1
    x = inputs[:left_index]
    y = [f_left(i, f_right_i, smoothed_data, inputs) for i in x]

    spectrum = data_train[0, :]
    flux = data_train[f_right_i, :]

    fig, ax = plt.subplots(figsize=(10, 8))
    # predicted
    ax.scatter(x, y, s=5, c='red', alpha=0.5)
    
    # original smoothed
    ax.scatter(inputs, smoothed_data[f_right_i, :], s=5, c='blue', alpha=0.5)
    
    # original raw data
    ax.scatter(spectrum, flux, s=5, c='black', alpha=0.25)
    plt.show()

# for training example 20
plot_f_left(20, smoothed_data, inputs, data_train)


# In[114]:


def training_error(smoothed_data, inputs):
    left_index = np.where(inputs == 1200)[0][0] - 1
    total_err = 0
    
    for f_right_i in range(smoothed_data.shape[0]):
        for x_eval in inputs[0:left_index]:
            y_pred = f_left(x_eval, f_right_i, smoothed_data, inputs)
            y_err = (y_pred - f(f_right_i, smoothed_data, x_eval)) ** 2
            total_err += y_err
        
    return total_err / smoothed_data.shape[0]

print(training_error(smoothed_data, inputs))


# In[113]:


def test_error(smoothed_data, inputs, data_test):
    smoothed_test_data, test_inputs = smooth_data(data_test, tau=5)
    left_index = np.where(inputs == 1200)[0][0] - 1
    total_err = 0
    
    for f_right_i in range(smoothed_test_data.shape[0]):
        for x_eval in inputs[0:left_index]:
            y_pred = f_left(x_eval, f_right_i, smoothed_test_data, inputs)
            y_err = (y_pred - f(f_right_i, smoothed_test_data, x_eval)) ** 2
            total_err += y_err
        
    return total_err / smoothed_test_data.shape[0]

print(test_error(smoothed_data, inputs, data_test))


# In[112]:


def plot_f_left_test(f_right_i, smoothed_data, inputs, data_test):
    smoothed_test_data, test_inputs = smooth_data(data_test, tau=5)
    left_index = np.where(inputs == 1200)[0][0] - 1
    
    line_y = []
    for x_eval in inputs[0:left_index]:
        y_pred = f_left(x_eval, f_right_i, smoothed_data, inputs)
        line_y.append(y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    # predicted
    ax.plot(inputs[0:left_index], line_y, lw=3, c='red', alpha=0.5)
    
    # actual
    ax.plot(test_inputs, smoothed_test_data[f_right_i, :], lw=3, c='blue', alpha=0.5)
    
    plt.show()

plot_f_left_test(6, smoothed_data, inputs, data_test)

