import random
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    return (labvitals_time_series_list_train, labvitals_time_series_list_test,
            labvitals_time_series_list_val, labels_train, labels_test, labels_val)


def get_time_series(data_frame, name="train"):
    """creates a list of time series for the given file.

    Args:
        data_frame (Pandas DataFrame) dataset as Pandas DataFrame
        name (str, optional): name of dataset for progress bar. Defaults to "train".

    Returns:
        List of time Series: a list of time series for each icustay_id
    """
    # fill missing values with mean of value before and after.
    # if first or last value is nan, just take the value after or before.
    data_frame = (data_frame.ffill()+data_frame.bfill())/2
    #data_frame = data_frame.interpolate()
    data_frame = data_frame.bfill().ffill()
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
    