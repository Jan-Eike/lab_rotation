import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from tslearn import metrics
from save_data import (save_classification_data,
                       delete_classification_data, 
                       save_current_test_data, 
                       save_nn_with_false_label)


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


def classify(labvitals_list_train, labvitals_list_test, labels_train, labels_test, best_k=5, print_res=True, save_classification=False):
    """classifies the test data wrt the given train data

    Args:
        labvitals_list_train (List of time Series): List of time series for training
        labvitals_list_test (List of time Series): List of time series for testing
        labels_train (List of Integers): training labels
        labels_test (List of Integers): testing labels
        best_k (int, optional): number of nearest neighbors. Defaults to 5.
        print_res (bool, optional): Print results or not. Defaults to True.
        save_classification (bool, optional): Save the results of the classification. Defaults to False.

    Returns:
        float: mean auprc score
        float: mean roc auc score
    """
    scores_auprc = []
    scores_roc_auc = []
    # delete old data if one wants to save new data
    if save_classification:
        delete_classification_data()

    score_auprc, score_roc_auc = knn(labvitals_list_train, labvitals_list_test, 
                                    labels_train, labels_test, k=best_k, save_classification=save_classification)

    scores_auprc.append(score_auprc)
    scores_roc_auc.append(score_roc_auc)

    if save_classification:
        save_current_test_data(labvitals_list_test)

    mean_score_auprc = np.mean(scores_auprc)
    mean_score_roc_auc = np.mean(scores_roc_auc)
    if print_res:
        print("Mean score_auprc:   {}".format(mean_score_auprc))
        print("Mean score_roc_auc: {}".format(mean_score_roc_auc))
    return mean_score_auprc, mean_score_roc_auc


def knn(time_series_list_train, time_series_list_test, labels_train, labels_test, k=5, save_classification=False):
    """perfomrs K nearest neighbors for the given train and test set.

    Args:
        time_series_list_train (List of time Series): List of time series for training
        time_series_list_test (List of time Series): List of time series for testing
        labels_train (List of Integers): training labels
        labels_test (List of Integers): testing labels
        k (int, optional): number of nearest neighbors. Defaults to 5.
        save_classification (bool, optional): Save the results of the classification. Defaults to False.

    Returns:
        float: auprc score
        float: roc auc score
    """
    pred_labels = []
    k_nearest_time_series = []
    # iloc[:, 6:] just cuts off the first 6 columns, since we don't need them for calculating anything
    number_of_channels = time_series_list_train[0].iloc[:, 6:].shape[1]

    num_cores = 2#multiprocessing.cpu_count() - 2
    parameters = (pred_labels, k_nearest_time_series, number_of_channels,
                  save_classification, k, labels_train, time_series_list_train)
    inputs = tqdm(time_series_list_test, desc="Claculating DTW distance for entire test data", leave=False, position=2)
    # parallel call for the method
    pred_labels = Parallel(n_jobs=num_cores)(delayed(test_parallel)(input, parameters, i) for i, input in enumerate(inputs))

    score_auprc = average_precision_score(labels_test, pred_labels)
    score_roc_auc = roc_auc_score(labels_test, pred_labels)
    """
    ray.init(log_to_driver=False)
    for i, input in enumerate(inputs):
        pred_labels.append(test_parallel.remote(input, parameters, i))
    
    score_auprc = average_precision_score(labels_test, pred_labels)
    score_roc_auc = roc_auc_score(labels_test, pred_labels)
    """

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
    return score_auprc, score_roc_auc


def test_parallel(time_series_test, parameters, i):
    pred_labels, k_nearest_time_series = parameters[0:2]
    number_of_channels, save_classification = parameters[2:4]
    k, labels_train, time_series_list_train = parameters[4:7]
    distances_per_test_point = []
    best_paths_per_test_point = []
    for time_series_train in tqdm(time_series_list_train, desc="Calculating DTW distance for one test point", leave=False):
        distances_per_train_point = []
        best_paths_per_train_point = []
        for channel in range(number_of_channels):
            # distance from one test point to all training points
            best_path, d = metrics.dtw_path(np.array(time_series_test.iloc[:, 6:].iloc[:, [channel]]), np.array(time_series_train.iloc[:, 6:].iloc[:, [channel]]))
            distances_per_train_point.append(d)
            best_paths_per_train_point.append(best_path)

        best_paths_per_test_point.append(best_paths_per_train_point)
        distances_per_test_point.append(distances_per_train_point)

    distances_per_test_point = np.array(distances_per_test_point)
    distances = np.mean(distances_per_test_point, axis=1)

    sorted_distances = np.argsort(distances)
    nearest_neighbor_id = sorted_distances[:k]

    if save_classification:
        save_classification_data((index_time_series_list(time_series_list_train, nearest_neighbor_id), i),
                                 (index_time_series_list(best_paths_per_test_point, nearest_neighbor_id), i), 
                                 (index_time_series_list(distances, nearest_neighbor_id), i),
                                 (index_time_series_list(distances_per_test_point, nearest_neighbor_id), i))

    pred_label = labels_train[nearest_neighbor_id]
    # finds most frequently occurring label 
    pred_label = np.bincount(pred_label).argmax()
    pred_labels.append(pred_label)

    find_nn_with_false_label(sorted_distances, labels_train, pred_label, time_series_list_train, save_classification)

    return pred_labels


def find_nn_with_false_label(sorted_distances, labels_train, pred_label, time_series_list_train, save_classification):
    """find the nearest neighbor that has a different label than the predicted one and
       saves it if required. Method is useless if the result does not get saved.

    Args:
        sorted_distances (List): List of indices of sorted distances
        labels_train (List): List of trianing labels
        pred_label (List): List of predicted Lbales
        time_series_list_train (List of time Series): List of time series for training
        save_classification (bool, optional): Save the results of the classification. Defaults to False.
    """
    if save_classification:
        i = 0
        dist = sorted_distances[i]
        while labels_train[dist] == pred_label:
            i += 1
            dist = sorted_distances[i]
        nn_with_false_label = time_series_list_train[dist]
        save_nn_with_false_label(nn_with_false_label)
