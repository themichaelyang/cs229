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
    # print(f1_y.shape, f2_y.shape)
    sq_dist = (f1_y - f2_y).T @ (f1_y - f2_y)
    # This was the problem: the distance function was
    # actually squaring it after it summed the diffs
    # Leaving this in for reference later.
    # sq_dist = (f1_y - f2_y).sum() ** 2
    # return sq_dist.sum()
    return sq_dist


def sq_distance_rows(rows, fixed):
    sq_dist_axis = lambda axis: sq_distance(axis, fixed)
    sq_dists = np.apply_along_axis(sq_dist_axis, 1, rows)
    return sq_dists


def ker(t):
    return max(1 - t, 0)


class Model:
    def __init__(self, Y_train, wavelengths):
        """
            Y_train is passed in as smoothed data here
            Y_train is a m by n matrix of many f() evaluated over
            all wavelengths (w/o absorption)
        """
        LYMAN_ALPHA = 1300
        LYMAN_FOREST_MAX = 1200

        self.wavelengths = wavelengths
        self.right_start = np.where(wavelengths == LYMAN_ALPHA)[0][0]
        self.left_end = np.where(wavelengths == LYMAN_FOREST_MAX)[0][0]

        # Split data for f_left and f_right
        # These Y values are the outputs of f (Y_train: absorption free data)
        self.Y_right_train = Y_train[:, self.right_start:]
        self.Y_left_train = Y_train[:, :self.left_end]

        # vectorized ker function
        self.v_ker = np.vectorize(ker)

    def predict(self, y_right_pred, k=3):
        """
            Predict y_left_pred from y_right_pred (which has absorption!)
            y_right_pred: (n - right_start) length vector, one f_right()
        """
        # sq_dist_row = lambda row: sq_distance(row, y_right_pred)
        # sq_dists = np.apply_along_axis(sq_dist_row, 0, self.Y_right_train)
        sq_dists = sq_distance_rows(self.Y_right_train, y_right_pred)

        # get sorted indicies
        ordered_indicies = np.argsort(sq_dists)

        knn_indicies = ordered_indicies[:k]
        max_dist_index = ordered_indicies[-1]

        # get knn distances via indicies
        knn_dists = sq_dists[knn_indicies]
        h = sq_dists[max_dist_index]

        y_left_pred = deque()
        wavelengths_left = self.wavelengths[:self.left_end]

        for wv in wavelengths_left:
            y_i_pred = self.predict_pt(wv, y_right_pred, knn_indicies, knn_dists, h)
            y_left_pred.append(y_i_pred)

        return wavelengths_left, y_left_pred


    def predict_pt(self, x_pred, y_right_pred, knn_indicies, knn_dists, h):
        # get all outputs of training f_left(x_pred) for knn
        knn_left_i = knn_indicies
        knn_left_col = self.wavelength_to_index(x_pred)
        knn_y_left = (self.Y_left_train[:, knn_left_col])[knn_left_i]

        numerator = (self.v_ker(knn_dists / h) @ knn_y_left)
        denominator = self.v_ker(knn_dists / h).sum()

        return numerator / denominator


    def split_left_right(self, Y):
        return Y[:, :self.left_end], Y[:, self.right_start:]


    def wavelength_to_index(self, wv):
        return np.where(self.wavelengths == wv)[0][0]


def load_or_smooth(filename, X, Y, tau):
    Y_smooth = None
    if not os.path.exists(filename):
        Y_smooth = smooth(X, Y, tau)
        pickle.dump(Y_smooth, open(filename, 'wb'))
    else:
        Y_smooth = pickle.load(open(filename, 'rb'))
    return Y_smooth


def graph_sm(x_orig, y_orig,
          x_smooth, y_smooth):
    """
        Plot a single training set
    """
    fig, ax = plt.subplots()

    ax.scatter(x_orig, y_orig, c='black', s=3, alpha=0.2)
    ax.plot(x_smooth, y_smooth, c='red', lw=1)


def graph(x_orig, y_orig,
          x_smooth, y_smooth,
          x_pred, y_pred):
    """
        Plot a single training set
    """
    fig, ax = plt.subplots()

    # ax.scatter(x_orig, y_orig, c='black', s=3, alpha=0.2)
    ax.plot(x_smooth, y_smooth, c='black', lw=1)
    ax.plot(x_pred, y_pred, c='red', lw=1)


def main():
    TAU = 5

    data_file = "data/quasar_train.csv"
    data_test_file = "data/quasar_test.csv"

    data_train = np.genfromtxt(data_file, delimiter=",")
    data_test = np.genfromtxt(data_test_file, delimiter=",")

    X_test, Y_test = split_data(data_test)
    wavelengths, intensities = split_data(data_train)

    Y_train = intensities
    X_train = prepend_ones(wavelengths)
    X_test = prepend_ones(X_test)

    assert((X_train == X_test).all())

    # Get smoothed Y of training set
    Y_train_sm = load_or_smooth('Y_train_sm.pkl', X_train, Y_train, TAU)
    Y_test_sm = load_or_smooth('Y_test_sm.pkl', X_test, Y_test, TAU)


    model = Model(Y_train_sm, wavelengths)
    Y_test_left, Y_test_right = model.split_left_right(Y_test_sm)

    ith = 5
    x_i_test = wavelengths
    y_i_test = Y_test[ith, :]

    y_i_test_sm = Y_test_sm[ith, :]

    y_i_test_right = Y_test_right[ith, :]
    x_i_test_pred, y_i_test_pred = model.predict(y_i_test_right)

    graph_sm(wavelengths, Y_train[0, :],
             wavelengths, Y_train_sm[0, :])

    graph(x_i_test, y_i_test,
          x_i_test, y_i_test_sm,
          x_i_test_pred, y_i_test_pred)

    plt.show()


if __name__ == "__main__":
    main()
