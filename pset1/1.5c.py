import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def get_weights(x_eval, X, tau):
    """
        x_eval: n+1-vector
        X: m by n+1 array
        tau: bandwidth param

        Returns: diagonal matrix of weights
    """
    sum_sq_diff = ((X - x_eval) ** 2).sum(axis=1)
    constant = 1 / (2 * (tau**2))
    weights = np.exp(-sum_sq_diff * constant)
    W = np.diag(weights)
    return W


def evaluate(x_eval, theta_eval):
    return theta_eval @ x_eval


def pinv(mat):
    return np.linalg.pinv(mat) if mat.ndim > 1 else (1 / mat)


def lw_regression(x_eval, X, Y, tau):
    W = get_weights(x_eval, X, tau)
    theta_pred = pinv(X.T @ W @ X) @ X.T @ W @ Y
    return theta_pred


def prepend_ones(vec):
    return np.insert(vec.reshape(-1, 1), 0, values=1, axis=1)


def smooth(X, Y, tau=5):
    smoothed_Y = deque()
    for y_i in Y:
        smoothed_y_i = []
        for x_eval in X:
            theta_pred = lw_regression(x_eval, X, y_i, tau)
            smoothed_y_i.append(evaluate(theta_pred, x_eval))
        smoothed_Y.append(smoothed_y_i)

    return np.asarray(smoothed_Y)


def split_data(data):
    """
        wavelengths: 1 x n vector of lambda values (X)
        intensities: m x n vector of flux values (Y)
    """
    wavelengths = data[0, :]
    intensities = data[1:, :]

    return wavelengths, intensities


def main():
    TAU = 5

    data_file = "data/quasar_train.csv"
    data_test_file = "data/quasar_test.csv"

    data_train = np.genfromtxt(data_file, delimiter=",")
    data_test = np.genfromtxt(data_test_file, delimiter=",")

    wavelengths, intensities = split_data(data_train)

    Y_train = intensities
    X_train = prepend_ones(wavelengths)

    print(smooth(X_train, Y_train, TAU))




if __name__ == "__main__":
    main()
