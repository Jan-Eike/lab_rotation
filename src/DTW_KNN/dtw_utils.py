"""All these method are taken from tslearn.metrics
   One slight modification was done, to get access to
   the dtw matrix. I had to copy these methods
   because some of them were private.
"""
import pandas
import numpy as np
from numba import njit


@njit()
def local_squared_dist(x, y):
    dist = 0.
    for di in range(x.shape[0]):
        diff = (x[di] - y[di])
        dist += diff * diff
    return dist


@njit()
def get_matrix(s1, s2, mask):
    l1 = s1.shape[0]
    l2 = s2.shape[0]
    cum_sum = np.full((l1 + 1, l2 + 1), np.inf)
    cum_sum[0, 0] = 0.

    for i in range(l1):
        for j in range(l2):
            if np.isfinite(mask[i, j]):
                cum_sum[i + 1, j + 1] = local_squared_dist(s1[i], s2[j])
                cum_sum[i + 1, j + 1] += min(cum_sum[i, j + 1],
                                            cum_sum[i + 1, j],
                                            cum_sum[i, j])
    return cum_sum[1:, 1:]


@njit()
def return_path(acc_cost_mat):
    sz1, sz2 = acc_cost_mat.shape
    path = [(sz1 - 1, sz2 - 1)]
    while path[-1] != (0, 0):
        i, j = path[-1]
        if i == 0:
            path.append((0, j - 1))
        elif j == 0:
            path.append((i - 1, 0))
        else:
            arr = np.array([acc_cost_mat[i - 1][j - 1],
                               acc_cost_mat[i - 1][j],
                               acc_cost_mat[i][j - 1]])
            argmin = np.argmin(arr)
            if argmin == 0:
                path.append((i - 1, j - 1))
            elif argmin == 1:
                path.append((i - 1, j))
            else:
                path.append((i, j - 1))
    return path[::-1]


@njit()
def njit_dtw(s1, s2, mask):
    """Compute the dynamic time warping score between two time series.
    Parameters
    ----------
    s1 : array, shape = (sz1,)
        First time series.
    s2 : array, shape = (sz2,)
        Second time series
    mask : array, shape = (sz1, sz2)
        Mask. Unconsidered cells must have infinite values.
    Returns
    -------
    dtw_score : float
        Dynamic Time Warping score between both time series.
    """
    cum_sum = get_matrix(s1, s2, mask)
    return np.sqrt(cum_sum[-1, -1])


def dtw_path(s1, s2):
    s1 = to_time_series(s1)
    s2 = to_time_series(s2)
    mask = np.zeros((s1.shape[0], s2.shape[0]))
    acc_cost_mat = get_matrix(s1, s2, mask=mask)
    path = return_path(acc_cost_mat)
    return np.sqrt(acc_cost_mat[-1, -1])


def to_time_series(ts, remove_nans=False):
    ts_out = np.array(ts, copy=True)
    if ts_out.ndim <= 1:
        ts_out = ts_out.reshape((-1, 1))
    if ts_out.dtype != np.float:
        ts_out = ts_out.astype(np.float)
    if remove_nans:
        ts_out = ts_out[:ts_size(ts_out)]
    return ts_out


def ts_size(ts):
    ts_ = to_time_series(ts)
    sz = ts_.shape[0]
    while sz > 0 and np.all(np.isnan(ts_[sz - 1])):
        sz -= 1
    return sz