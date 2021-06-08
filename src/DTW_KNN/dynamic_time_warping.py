from tslearn.metrics import dtw
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm, trange
from sklearn.model_selection import KFold 


def main():
    """main method: Runs the entire clssification process."""
    # train, test and val length are just for testing purposes, to be able to
    # cut off parts of the datasets for faster computation
    train_length = -1
    test_length = -1
    val_length = -1
    data = load_data(train_length, test_length, val_length)
    labvitals_time_series_list_train, labels_train = data[0], data[3]
    labvitals_time_series_list_test, labels_test = data[1], data[4]
    labvitals_time_series_list_val, labels_val = data[2], data[5]
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    best_k = find_best_k(labvitals_time_series_list_val, labels_val, k_list)
    print("best k : {}".format(best_k))
    dtw_matrices_train, dtw_matrices_test = calculate_distance_matrices(labvitals_time_series_list_train, labvitals_time_series_list_test, train_length, test_length)
    classify(dtw_matrices_train, dtw_matrices_test, labels_train, labels_test, test_length, best_k=best_k)


def find_best_k(labvitals_time_series_list_val, labels_val, k_list):
    """finds the best parameter k for knn.

    Args:
        labvitals_time_series_list_val (list of time Series): list of time series for validation set
        labels_val (List of integers): list of labels
        k_list (list of integers): List containing each value k that is going to be checked

    Returns:
        Integer: the parameter k that yielded the best result
    """
    avg_scores = []
    for k in tqdm(k_list, desc="finding best k"):
        avg_score = cross_validate(labvitals_time_series_list_val, labels_val, k)
        avg_scores.append(avg_score)
    max_index = np.argmax(avg_scores)
    return k_list[max_index]


def cross_validate(labvitals_time_series_list_val, labels_val, k):
    """cross validation to compute an average score for the paramter k, using the validation set

    Args:
        labvitals_time_series_list_val (List of time Series): list of time series for validation set
        labels_val (List of integers): List of labels
        k (integer): parameter k that gets cross validated

    Returns:
        float: mean score of all scores achieved during cross validation
    """
    kf = KFold(n_splits=3, random_state=42, shuffle=True)
    scores = []
    # just defining a progress bar for manual control
    pbar = tqdm(total = 3, desc="Cross validating for k = {}".format(k), leave=False)
    for train_index , test_index in kf.split(labvitals_time_series_list_val):
        X_train = index_time_series_list(labvitals_time_series_list_val, train_index)
        X_test = index_time_series_list(labvitals_time_series_list_val, test_index)
        y_train , y_test = labels_val[train_index] , labels_val[test_index]
        dtw_matrices_train, dtw_matrices_test = calculate_distance_matrices(X_train, X_test, len(X_train), len(X_test))
        score = classify(dtw_matrices_train, dtw_matrices_test, y_train, y_test, len(X_test), best_k=k, print_res=False)
        scores.append(score)
        pbar.update(1)
    pbar.close()
    return np.mean(scores)


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


def calculate_distance_matrices(labvitals_time_series_list_train, labvitals_time_series_list_test, train_length, test_length):
    """calculates dtw-distance matrices for both train and test set

    Args:
        labvitals_time_series_list_train (List of time Series): time series for training
        labvitals_time_series_list_test (List of time Series): time Series for testing
        train_length (int): length of training dataset (just for testing purposes)
        test_length (int): length of test dataset (just for testing purposes)

    Returns:
        both lists of dtw-distance matrices
    """
    dtw_matrices_train = dtw_distance_per_channel(labvitals_time_series_list_train[:train_length], labvitals_time_series_list_train[:train_length], name="train")
    dtw_matrices_test = dtw_distance_per_channel(labvitals_time_series_list_train[:train_length], labvitals_time_series_list_test[:test_length], name="test")
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
                dynamic_time_warping_distance_matrix[i,j] = dtw(time_series_1.iloc[:, 6:].iloc[:, channel], time_series_2.iloc[:, 6:].iloc[:, channel])
        dynamic_time_warping_distance_matrices.append(dynamic_time_warping_distance_matrix)
    return dynamic_time_warping_distance_matrices


def classify(dtw_matrices_train, dtw_matrices_test, labels_train, labels_test, test_length, best_k=4, print_res=True):
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
        float: mean score of the classification
    """
    # use 'precomputed' as metrix to plug in a distance matrix instead of a set of points in some space (point matrix).
    nbrs = KNeighborsClassifier(n_neighbors=best_k, metric='precomputed')
    scores = []
    for i, dtw_matrix in enumerate(dtw_matrices_train):
        nbrs.fit(dtw_matrix, labels_train)
        score = nbrs.score(dtw_matrices_test[i].reshape(test_length,-1), labels_test)
        scores.append(score)
        if print_res:
            print("Channel {} prediction:  {}".format(i, nbrs.predict(dtw_matrices_test[i].reshape(test_length,-1))))
            print("Channel {} true labels: {}".format(i, labels_test))
            print("Channel {} score:       {}".format(i, score))
    mean_score = np.mean(scores)
    if print_res:
        print("Mean score: {}".format(mean_score))
    return mean_score


if __name__ == "__main__":
    main()
