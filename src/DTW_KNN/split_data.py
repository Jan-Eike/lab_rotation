from classification import load_data
import pickle
import numpy as np
from tslearn import metrics
from dtaidistance import dtw_visualisation as dtwvis
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = load_data(-1, -1, -1)
    labvitals_time_series_list_train, labels_train = data[0], data[3]
    labvitals_time_series_list_test, labels_test = data[1], data[4]
    labvitals_time_series_list_val, labels_val = data[2], data[5]
    mask = [i == 3 or i >= 6 for i in range(30)]
    l1 = labvitals_time_series_list_train[0].iloc[:, mask]
    print(l1)
    