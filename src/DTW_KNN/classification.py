import multiprocessing
import time
from typing import ContextManager
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import Pool
from contextlib import contextmanager
from operator import itemgetter
from sklearn.metrics import average_precision_score, roc_auc_score
from dtw_utils import dtw_path
from load_data import load_predicted_labels
from save_data import (save_classification_data,
                       delete_classification_data,
                       save_current_test_data,
                       save_nn_with_false_label,
                       save_predicted_labels,
                       delete_predicted_labels)


def index_time_series_list(time_series_list, index):
    """indexes a python list, using a numpy array

    Args:
        time_series_list (List of time Series)
        index (numpy array)

    Returns:
        List of time Series: new time series list, indexed by the index array
    """
    new_time_series_list = []
    for i in index:
        new_time_series_list.append(time_series_list[i])
    return new_time_series_list


def classify(labvitals_list_train, labvitals_list_test, labels_train, labels_test, pool, 
             best_k=5, print_res=True, save_classification=False, num_cores=-1):
    """classifies the test data wrt the given train data

    Args:
        labvitals_list_train (List of time Series): List of time series for training
        labvitals_list_test (List of time Series): List of time series for testing
        labels_train (List of Integers): training labels
        labels_test (List of Integers): testing labels
        best_k (int, optional): number of nearest neighbors. Defaults to 5.
        print_res (bool, optional): Print results or not. Defaults to True.
        save_classification (bool, optional): Save the results of the classification.
                                              Defaults to False.
        sum_cores (int, optional): Number of cores for multitasking. Defaults to -1.

    Returns:
        float: mean auprc score
        float: mean roc auc score
    """
    # delete old data if one wants to save new data
    if save_classification:
        delete_classification_data()

    # always delete this, needed for parallelization
    delete_predicted_labels()
    knn(labvitals_list_train, labvitals_list_test,
        labels_train, labels_test, pool, k=best_k,
        save_classification=save_classification, num_cores=num_cores)

    # get predicted labels from database (they were saved in knn method)
    # resort them and only extract the labels and not the indices.
    pred_labels = load_predicted_labels()
    pred_labels = sorted(pred_labels, key=itemgetter(1))
    pred_labels = list(map(list, zip(*pred_labels)))[0]

    print(labels_test)
    print(pred_labels)

    mean_score_auprc = average_precision_score(labels_test, pred_labels)
    mean_score_roc_auc = roc_auc_score(labels_test, pred_labels)

    if save_classification:
        save_current_test_data(labvitals_list_test)

    if print_res:
        print("Mean score_auprc:   {}".format(mean_score_auprc))
        print("Mean score_roc_auc: {}".format(mean_score_roc_auc))
    return mean_score_auprc, mean_score_roc_auc


def knn(time_series_list_train, time_series_list_test, labels_train, labels_test, pool,
        k=5, save_classification=False, num_cores=-1):
    """perfomrs K nearest neighbors for the given train and test set.

    Args:
        time_series_list_train (List of time Series): List of time series for training
        time_series_list_test (List of time Series): List of time series for testing
        labels_train (List of Integers): training labels
        labels_test (List of Integers): testing labels
        k (int, optional): number of nearest neighbors. Defaults to 5.
        save_classification (bool, optional): Save the results of the classification.
                                              Defaults to False.

    Returns:
        float: auprc score
        float: roc auc score
    """
    pred_labels = []
    # iloc[:, 6:] just cuts off the first 6 columns,
    # since we don't need them for calculating anything
    number_of_channels = time_series_list_train[0].iloc[:, 6:].shape[1]

    parameters = (pred_labels, number_of_channels, save_classification,
                  k, labels_train, time_series_list_train)

    # create progress bar for time_series_list_test
    inputs = tqdm(time_series_list_test, desc="Claculating DTW distance for entire test data",
                 leave=True, position=2)

    # parallel call for the predict method
    """
    Parallel(n_jobs=num_cores, backend="loky")(delayed(predict)(input, parameters, i)
                                    for i, input in enumerate(inputs))
    """
    with poolcontext(processes=num_cores) as pool:
        r = pool.map_async(predict_unpack, [(input, parameters, i) for i, input in enumerate(inputs)])
        r.wait()
    # k_nearest_time_series:
    # [[nn_1, ..., nn_k], ..., [nn_1, ..., nn_k]]
    # one list for each test point

    # best_paths:
    # [ [[(), ..., ()], ..., [(), ..., ()]], ..., [[(), ..., ()], ..., [(), ..., ()]] ]
    # one list for each test point
    # one 1. inner list for each channel
    # one 2. inner list for each k nearest neighbor
    # each 2. inner list is a DTW path

    # best_distances:
    # [[], ..., []]


def predict(time_series_test, parameters, i):
    """predict the labels for the given test and train data

    Args:
        time_series_test (Pandas time Series): one test time series
        parameters (Tuple): Tuple of the other parameters:
                            pred_labels (List): List of the predicted labels
                            number_of_channels (int): Number of channels (labvitals)
                            save_classification (bool): Save classification data or not
                            k (int): k for k nearest neighbors
                            labels_train (List): List of training labels
                            time_series_list_train (List): List of training time series
        i (int): number of current test point to restore correct order in the end

    Returns:
        List: predicted labels (each test point gets appened and then
              the new list will be passed to this method again)
    """
    pred_labels, number_of_channels = parameters[0:2]
    save_classification, k = parameters[2:4]
    labels_train, time_series_list_train = parameters[4:6]
    distances_per_test_point = []
    best_paths_per_test_point = []
    dtw_matrices_per_test_point = []
    for time_series_train in tqdm(time_series_list_train, desc="Calculating DTW distance for one test point", leave=False):
        distances_per_train_point = []
        best_paths_per_train_point = []
        dtw_matrices_per_train_point = []
        for channel in range(number_of_channels):
            # distance from one test point to all training points
            s1 = np.array(time_series_test.iloc[:, 6:].iloc[:, [channel]]).copy()
            s2 = np.array(time_series_train.iloc[:, 6:].iloc[:, [channel]]).copy()
            if s1.shape[0] != 0 and s2.shape[0] != 0:
                best_path, dist, dtw_matrix = dtw_path(s1,s2)
                
            distances_per_train_point.append(dist)
            best_paths_per_train_point.append(best_path)
            dtw_matrices_per_train_point.append(dtw_matrix)

        best_paths_per_test_point.append(best_paths_per_train_point)
        distances_per_test_point.append(distances_per_train_point)
        dtw_matrices_per_test_point.append(dtw_matrices_per_train_point)

        del distances_per_train_point, best_paths_per_train_point, dtw_matrices_per_train_point, best_path, dist

    distances_per_test_point = np.array(distances_per_test_point)
    distances = np.mean(distances_per_test_point, axis=1)

    sorted_distances = np.argsort(distances)
    nearest_neighbor_id = sorted_distances[:k]

    if save_classification:
        save_classification_data((index_time_series_list(time_series_list_train, nearest_neighbor_id), i),
                                 (index_time_series_list(best_paths_per_test_point, nearest_neighbor_id), i),
                                 (index_time_series_list(distances, nearest_neighbor_id), i),
                                 (index_time_series_list(distances_per_test_point, nearest_neighbor_id), i),
                                 (index_time_series_list(dtw_matrices_per_test_point, nearest_neighbor_id), i))

    pred_label = labels_train[nearest_neighbor_id]
    # finds most frequently occurring label
    pred_label = np.bincount(pred_label).argmax()

    # this will always be saved since it is needed for parallelization
    save_predicted_labels((pred_label, i))

    find_nn_with_false_label(sorted_distances, labels_train, pred_label,
                             time_series_list_train, save_classification)

    del time_series_list_train, best_paths_per_test_point, distances, distances_per_test_point, pred_label

    #return pred_labels


def predict_unpack(args):
    return predict(*args)


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        time.sleep(10)
        pool.terminate()
        pool.join()
        pool.close()

def find_nn_with_false_label(sorted_distances, labels_train, pred_label,
                             time_series_list_train, save_classification):
    """find the nearest neighbor that has a different label than the predicted one and
       saves it if required. Method is useless if the result does not get saved.

    Args:
        sorted_distances (List): List of indices of sorted distances
        labels_train (List): List of trianing labels
        pred_label (List): List of predicted Lbales
        time_series_list_train (List of time Series): List of time series for training
        save_classification (bool, optional): Save the results of the classification.
                                              Defaults to False.
    """
    if save_classification:
        i = 0
        dist = sorted_distances[i]
        while labels_train[dist] == pred_label:
            i += 1
            dist = sorted_distances[i]
        nn_with_false_label = time_series_list_train[dist]
        save_nn_with_false_label(nn_with_false_label, true_label=pred_label, false_label=labels_train[dist])
