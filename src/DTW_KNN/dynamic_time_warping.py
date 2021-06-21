from tslearn.metrics import dtw
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm, trange
from sklearn.metrics import average_precision_score, roc_auc_score
from save_data import save_distance_matrices


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
    labvitals_time_series_list_train = get_time_series(labvitals_train, name="train")[:train_length]
    labvitals_time_series_list_test = get_time_series(labvitals_test, name="test")[:test_length]
    labvitals_time_series_list_val = get_time_series(labvitals_val, name="val")[:val_length]
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


def classify(labvitals_list_train, labvitals_list_test, labels_train, labels_test, best_k=5, print_res=True):
    """classifies the test data wrt the given train data

    Args:
        labvitals_list_train (List of time Series): List of time series for training
        labvitals_list_test (List of time Series): List of time series for testing
        labels_train (List of Integers): training labels
        labels_test (List of Integers): testing labels
        best_k (int, optional): number of nearest neighbors. Defaults to 5.
        print_res (bool, optional): Print results or not. Defaults to True.

    Returns:
        float: mean auprc score
        float: mean roc auc score
    """
    number_of_channels = labvitals_list_train[0].iloc[:, 6:].shape[1]
    scores_auprc = []
    scores_auc = []
    for channel in trange(number_of_channels, desc="Classify channel"):
        score_auprc, score_roc_auc, pred_labels = knn(labvitals_list_train, labvitals_list_test, labels_train, labels_test, channel, k=best_k)
        scores_auprc.append(score_auprc)
        scores_auc.append(score_roc_auc)
    mean_score_auprc = np.mean(scores_auprc)
    mean_score_roc_auc = np.mean(scores_auc)
    if print_res:
        print("Mean score_auprc: {}".format(mean_score_auprc))
        print("Mean score_roc_auc:   {}".format(mean_score_roc_auc))
    return mean_score_auprc, mean_score_roc_auc


def knn(time_series_list_train, time_series_list_test, labels_train, labels_test, channel, k=5):
    """perfomrs K nearest neighbors for the given train and test set.

    Args:
        time_series_list_train (List of time Series): List of time series for training
        time_series_list_test (List of time Series): List of time series for testing
        labels_train (List of Integers): training labels
        labels_test (List of Integers): testing labels
        channel (Integer): current channel of the time series
        k (int, optional): number of nearest neighbors. Defaults to 5.

    Returns:
        float: auprc score
        float: roc auc score
        List of Integers: List of predicted labels
    """
    pred_labels = []
    for i, time_series_test in enumerate(tqdm(time_series_list_test, desc="Calculating DTW distance from test data to train data", leave=False)):
        distances = []
        for time_series_train in tqdm(time_series_list_train, desc="Calculating DTW distance to test Point {}".format(i), leave=False):
            distances.append(dtw(time_series_test.iloc[:, 6:].iloc[:, channel], time_series_train.iloc[:, 6:].iloc[:, channel]))

        nearest_neighbor_id = np.argsort(distances)[:k]
        pred_label = labels_train[nearest_neighbor_id]
        pred_label = np.bincount(pred_label).argmax()
        pred_labels.append(pred_label)

    score_auprc = average_precision_score(labels_test, pred_labels)
    score_roc_auc = roc_auc_score(labels_test, pred_labels)

    return score_auprc, score_roc_auc, pred_labels