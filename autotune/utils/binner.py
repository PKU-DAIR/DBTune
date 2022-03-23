import numpy as np

class Bin():

    def __init__(self, bin_start):
        self.deciles = None
        self.bin_start_ = bin_start

    def fit(self, matrix):
        self.deciles = []
        for col in matrix.T:
            self.deciles.append(get_deciles(col))
        return self

    def transform(self, matrix, copy=True):
        columns = []
        for col, decile in zip(matrix.T, self.deciles):
            columns.append(bin_by_decile(col, decile,
                                         self.bin_start_))
        res = np.vstack(columns).T
        return res

    def inverse_transform(self, matrix, copy=True):
        raise NotImplementedError("This method is not supported")


def get_deciles(y):
    decile_range = np.arange(10, 101, 10)
    deciles = np.percentile(y, decile_range)
    deciles[-1] = np.Inf
    return deciles


def bin_by_decile(matrix, deciles, bin_start):
    binned_matrix = np.zeros_like(matrix)
    for i in range(10)[::-1]:
        decile = deciles[i]
        binned_matrix[matrix <= decile] = i + bin_start
    return binned_matrix
