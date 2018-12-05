import numpy as np
import matplotlib.pyplot as plt

from collections import deque

import pickle
import os

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


def regression(X, Y, W=None):
    if W is None:
        m = X.shape[0]
        W = np.eye(m)
    theta_pred = pinv(X.T @ W @ X) @ X.T @ W @ Y
    return theta_pred


def lw_regression(x_eval, X, Y, tau):
    W = get_weights(x_eval, X, tau)
    return regression(X, Y, W)
    # theta_pred = pinv(X.T @ W @ X) @ X.T @ W @ Y
    # return theta_pred


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


def sq_distance(f1_y, f2_y):
    dist = (f1_y - f2_y)
    return dist @ dist


def ker(t):
    return max(1 - t, 0)


class Model:
    def __init__(Y_train, wavelengths):
        """ Y_train is passed in as smoothed data here """
        LYMAN_ALPHA = 1300
        LYMAN_FOREST_MAX = 1200

        self.wavelengths = wavelengths
        self.right_start = np.where(wavelengths == LYMAN_ALPHA)[0][0]
        self.left_end = np.where(wavelengths == LYMAN_FOREST_MAX)[0][0]

        # Split data for f_left and f_right
        # These Y values are the outputs of f (Y_train: absorption free data)
        self.Y_train_right = Y_train[:, self.right_start:]
        self.Y_train_left = Y_train[:, :self.left_end]


    def predict(Y_right_pred):
        """ Predict Y_left_pred from Y_right_pred (which has absorption) """
        wavelengths_left = self.wavelengths[:self.left_end]
        Y_left_pred = deque()

        dists = sq_distance()

        for wv in wavelengths_left:
            y_i_pred = self.predict_pt(wv, Y_right_pred)
            Y_left_pred.append()

        return wavelengths_left, Y_left_pred

    def predict_pt(x_pred, Y_right_pred):
        pass


def main():
    TAU = 5

    data_file = "data/quasar_train.csv"
    data_test_file = "data/quasar_test.csv"

    data_train = np.genfromtxt(data_file, delimiter=",")
    data_test = np.genfromtxt(data_test_file, delimiter=",")

    wavelengths, intensities = split_data(data_train)

    Y_train = intensities
    X_train = prepend_ones(wavelengths)

    # Get smoothed Y of training set
    Y_train_sm_file = 'Y_train_sm.pkl'
    Y_train_sm = None

    if not os.path.exists(Y_train_sm_file):
        Y_train_sm = smooth(X_train, Y_train, TAU)
        pickle.dump(Y_train_sm, open(Y_train_sm_file, 'wb'))
    else:
        Y_train_sm = pickle.load(open(Y_train_sm_file, 'rb'))


if __name__ == "__main__":
    main()
