from inspect import Parameter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import multiprocessing
from joblib import Parallel, delayed
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, roc_auc_score
from dtaidistance import dtw
from tslearn import metrics
from save_data import (save_distance_matrices,
                       save_classification_data,
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


def load_data(train_length, test_length, val_length):
    """loads data from the csv files, gets the time series from that data
       and gets the labels out of that data.

    Args:
        train_length (int): lnegth of the training dataset (just for testing purposes)
        test_length (int): length of the test dataset (just for testing purposes)
        val_length (int): length of the validation dataset (just for testing purposes)

    Returns:
        all the loaded data and labels
    """
    # We extract our data from the full_labvitals.csv dataset.
    # labvitals are both vital parameters and laboratory parameters, 
    # as one can see in Table A.1 in Moor et al.
    labvitals_train = pd.read_csv("C:\\Users\\Jan\\Desktop\\project\\MGP-AttTCN\\data\\train\\full_labvitals.csv")
    labvitals_test = pd.read_csv("C:\\Users\\Jan\\Desktop\\project\\MGP-AttTCN\\data\\test\\full_labvitals.csv")
    labvitals_val = pd.read_csv("C:\\Users\\Jan\\Desktop\\project\\MGP-AttTCN\\data\\val\\full_labvitals.csv")
    random.seed(10)
    if train_length != -1:
        labvitals_time_series_list_train = random.sample(get_time_series(labvitals_train, name="train"), train_length)
    else:
        labvitals_time_series_list_train = get_time_series(labvitals_train, name="train")
    if test_length != -1:
        labvitals_time_series_list_test = random.sample(get_time_series(labvitals_test, name="test"), test_length)
    else:
        labvitals_time_series_list_test = get_time_series(labvitals_test, name="test")
    if val_length != -1:
        labvitals_time_series_list_val = random.sample(get_time_series(labvitals_val, name="val"), val_length)
    else:
        labvitals_time_series_list_val = get_time_series(labvitals_val, name="val")
    labels_train = get_labels(labvitals_time_series_list_train)
    labels_test = get_labels(labvitals_time_series_list_test)
    labels_val = get_labels(labvitals_time_series_list_val)
    return (labvitals_time_series_list_train, labvitals_time_series_list_test,labvitals_time_series_list_val, labels_train, labels_test, labels_val)


def get_time_series(data_frame, name="train"):
    """creates a list of time series for the given file.
       each missing value gets filled up with 0 since it is assumed to be 
       within a healthy norm, as discussed in Rosnati et al.

    Args:
        data_frame (Pandas DataFrame) dataset as Pandas DataFrame
        name (str, optional): name of dataset for progress bar. Defaults to "train".

    Returns:
        List of time Series: a list of time series for each icustay_id
    """
    # fill missing values with 0
    data_frame = data_frame.fillna(0)
    # get a list of unique icustay_ids 
    icustay_ids = set(list(data_frame["icustay_id"]))
    labvitals_time_series_list = []
    for icustay_id in tqdm(icustay_ids, desc="creating time series for {} dataset".format(name)):
        labvitals_time_series = data_frame[data_frame["icustay_id"] == icustay_id]
        labvitals_time_series_list.append(labvitals_time_series)
    print()
    return labvitals_time_series_list


def get_labels(labvitals_time_series_list):
    """gets the label for each time series in the time series list.
       I decided to take the label 1 if the label is 1 at atleast 1
       point in the time series.

    Args:
        labvitals_time_series_list (List of time Series)

    Returns:
        List of Integers: list of labels for each patient in 
                          the time series list
    """
    labels = np.zeros((len(labvitals_time_series_list)), dtype=int)
    for i, time_series in enumerate(labvitals_time_series_list):
        for label in time_series["label"]:
            if label == 0:
                labels[i] = 0
            if label == 1:
                labels[i] = 1
                break
    return labels


# not used anymore
def calculate_distance_matrices(labvitals_time_series_list_train, labvitals_time_series_list_test, train_length, test_length, save=False, test_only=False):
    """calculates dtw-distance matrices for both train and test set

    Args:
        labvitals_time_series_list_train (List of time Series): time series for training
        labvitals_time_series_list_test (List of time Series): time Series for testing
        train_length (int): length of training dataset (just for testing purposes)
        test_length (int): length of test dataset (just for testing purposes)
        save (bool, optional): Save matrices or not. Defaults to False.
        test_only (bool, optional): Calculate test matrices only. Defaults to False.

    Returns:
        both lists of dtw-distance matrices
    """
    if test_only:
        dtw_matrices_test = dtw_distance_per_channel(labvitals_time_series_list_train[:train_length], labvitals_time_series_list_test[:test_length], name="test")
        return dtw_matrices_test
    dtw_matrices_train = dtw_distance_per_channel(labvitals_time_series_list_train[:train_length], labvitals_time_series_list_train[:train_length], name="train")
    dtw_matrices_test = dtw_distance_per_channel(labvitals_time_series_list_train[:train_length], labvitals_time_series_list_test[:test_length], name="test")
    if save:
        save_distance_matrices(dtw_matrices_train, "dtw_matrices_train")
    return dtw_matrices_train, dtw_matrices_test


# not used anymore
def dtw_distance_per_channel(labvitals_time_series_list_1, labvitals_time_series_list_2, name="train"):
    """calculates the dynamic time warping distance between each time series from
       the first list and the second list. For each time Series, the distance between
       each channel (column/dimesnion) gets calculated seperately.

    Args:
        labvitals_time_series_list_1 (List of time Series)
        labvitals_time_series_list_2 (List of time Series)
        name (str, optional): name of dataset for progress bar. Defaults to "train".

    Returns:
        List of Matrices: List of dynamic timewarping distance matrices. Each matrix
                          corresponds to the distance regarding one dimension (column)
    """
    N = len(labvitals_time_series_list_1)
    M = len(labvitals_time_series_list_2)
    dynamic_time_warping_distance_matrices = []
    # iloc[:, 6:] just cuts off the first 6 columns, since we don't need them for calculating anything
    number_of_channels = labvitals_time_series_list_1[0].iloc[:, 6:].shape[1]
    for channel in trange(number_of_channels, desc="claculating dtw-distance for {} dataset".format(name), leave=False):
        dynamic_time_warping_distance_matrix = np.zeros((N,M))
        for i, time_series_1 in enumerate(tqdm(labvitals_time_series_list_1, desc="claculating dtw-distance for one channel", leave=False)):
            for j, time_series_2 in enumerate(labvitals_time_series_list_2):
                # iloc[:, channel] takes the entire column with the number channel
                dynamic_time_warping_distance_matrix[i,j] = dtw(time_series_1.iloc[:, 6:].iloc[:, channel], time_series_2.iloc[:, 6:].iloc[:, channel])
        dynamic_time_warping_distance_matrices.append(dynamic_time_warping_distance_matrix)
    return dynamic_time_warping_distance_matrices


# not used anymore
def classify_precomputed(dtw_matrices_train, dtw_matrices_test, labels_train, labels_test, test_length, best_k=4, print_res=True):
    """
    classifies the data

    Args:
        dtw_matrices_train (list of matrices): List of matrices containing the per channel distance for the training set
        dtw_matrices_test (list of matrices): List of matrices containing the per channel distance for the test set
        labels_train (list of integers): training labels
        labels_test (list of integers): test labels
        test_length (int): length of test dataset (just for testing purposes)
        best_k (int, optional): best k for knn from cross validation. Defaults to 4.
        print_res (bool, optional): True if you want to print result, otherwise false. Defaults to True.
    
    Returns:
        float: mean acc_score of the classification
        float: mean auprc_score of the classification
    """
    # use 'precomputed' as metric to plug in a distance matrix instead of a set of points in some space (point matrix).
    nbrs = KNeighborsClassifier(n_neighbors=best_k, metric='precomputed')
    scores_acc = []
    scores_auprc = []
    for i, dtw_matrix in enumerate(dtw_matrices_train):
        nbrs.fit(dtw_matrix, labels_train)
        pred_labels = nbrs.predict(dtw_matrices_test[i].reshape(test_length,-1))
        score_auprc = average_precision_score(labels_test, pred_labels)
        scores_auprc.append(score_auprc)
        score_acc = nbrs.score(dtw_matrices_test[i].reshape(test_length,-1), labels_test)
        scores_acc.append(score_acc)
        if print_res:
            print("Channel {} prediction:  {}".format(i, pred_labels))
            print("Channel {} true labels: {}".format(i, labels_test))
            print("Channel {} score_acc:   {}".format(i, score_acc))
            print("Channel {} score_auprc: {}".format(i, score_auprc))
    mean_score_acc = np.mean(scores_acc)
    mean_score_auprc = np.mean(scores_auprc)
    if print_res:
        print("Mean score_acc:   {}".format(mean_score_acc))
        print("Mean score_auprc: {}".format(mean_score_auprc))
    return mean_score_acc, mean_score_auprc


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

    num_cores = min(20, multiprocessing.cpu_count() - 2)
    parameters = (pred_labels, k_nearest_time_series, number_of_channels,
                  save_classification, k, labels_train, time_series_list_train)
    inputs = tqdm(time_series_list_test, position=2, desc="Claculating DTW distance for entire test data", leave=False)
    multiprocessing.freeze_support()
    # parallel call for the method
    pred_labels = Parallel(n_jobs=num_cores, backend="multiprocessing")(delayed(test_parallel)(input, parameters, i) for i, input in enumerate(inputs))

    score_auprc = average_precision_score(labels_test, pred_labels)
    score_roc_auc = roc_auc_score(labels_test, pred_labels)

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


def plot_time_series(k_nearest_time_series, number_of_channels):
    plt.style.use('seaborn')
    for i, time_series in enumerate(k_nearest_time_series):
        print()
        print("Time Series {}:\n {}".format(i, time_series[0].iloc[:, 6:]))
        print()
        sqrtn = int(np.ceil(np.sqrt(number_of_channels)))
        fig = plt.figure(figsize=(sqrtn*2, sqrtn*2), dpi=300)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.5, hspace=0.9)
        sqrtn1 = sqrtn
        sqrtn2 = sqrtn
        # don't want the figure to be too wide
        if sqrtn >= 5:
            sqrtn1 = 4
            sqrtn2 = int(np.ceil((sqrtn * sqrtn) / sqrtn1))
        gs = gridspec.GridSpec(sqrtn2, sqrtn1)
        for i, channel_p in enumerate(time_series[0].iloc[:, 6:]):
            ax = plt.subplot(gs[i])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            xvalue = time_series[0]["chart_time"]
            yvalue = time_series[0].iloc[:, 6:][channel_p]
            ax.plot(xvalue, yvalue, linewidth=0.5, marker="s", markersize=1.5)
            #ax.set_xticklabels(xvalue, fontsize=5)
            #ax.set_yticklabels(yvalue, fontsize=5)
            ax.set_title(channel_p, fontsize=5)
        plt.show()
