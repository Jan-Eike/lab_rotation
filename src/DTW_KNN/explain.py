from save_data import load_classification_data, load_current_test_data, load_nn_with_false_label
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_visualisation as dtwvis
from tslearn import metrics


def main():
    """Main method for explainabilty.
    """
    k_nearest_time_series, best_paths, best_distances, distances_per_test_point  = load_classification_data()
    labvitals_list_test = load_current_test_data()
    nn_with_false_label = load_nn_with_false_label()
    #print(nn_with_false_label)
    plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, best_distances)


def plot_explain(nearest_neighbor, labvitals_list_test, best_paths, nn, test_point):
    """plots the anligned dynamic time warping between each channel of two time series.

    Args:
        nearest_neighbor (DataFrame): the nearest neighbor DataFrame
        labvitals_list_test (List of DataFrames): List of the test DataFrames
        best_paths (List): List of the best dtw paths for each test point
        nn (int): number of the nearest neighbor
        test_point (int): number of the test point
    """
    number_of_channels = labvitals_list_test[test_point].iloc[:, 6:].shape[1]
    plot_dtw = True
    for channel in range(number_of_channels-20):
        # the first indexing number stands for the current test point when this will be done automatically in the end (or not idk)

        # labvitals_list_test[test_point] takes the test_point-th DataFrame from the list
        # iloc[:, 6:] cuts off the first 5 unnecessary columns
        # iloc[:, [channel]] takes the column with the index channel
        # np.array transforms it to a numpy array
        # reshape(-1,) reshapes it from (a,1) to (a,) (from 2 dimensions to 1)
        time_series_1 = np.array(nearest_neighbor.iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
        time_series_2 = np.array(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
        #print(nearest_neighbor.iloc[:, 6:].iloc[:, [channel]])
        #print(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]])
        if plot_dtw:
            fig, ax = dtwvis.plot_warping(time_series_2, time_series_1, best_paths[test_point][0][nn][channel])
            fig.set_size_inches(10, 5)
            fig.subplots_adjust(hspace=0.2)
            fig.suptitle(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]].columns[0], y=1)
            ax[0].set_title("test data time series", y=1, pad=-14)
            ax[1].set_title("nearest neighbor", y=1, pad=-14)
            plt.show()


def plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, best_distances):
    """calls plot_explain for each test point in the test_point list
       and prints out the best distance from each test point to its k nearest neighbors.

    Args:
        k_nearest_time_series (List): List of the DataFrames of the k nearest neighbors
        labvitals_list_test (List of DataFrames): List of the test DataFrames
        best_paths (List): List of the best dtw paths for each test point
        best_distances (List): List of the smallest distances for each Test point
    """
    test_points = [0, 1, 2, 3, 4, 5, 6, 7]
    for test_point in test_points:
        print("Test point {}:".format(test_point))
        for k, nearest_neighbor in enumerate(k_nearest_time_series[test_point][0]):
            plot_explain(nearest_neighbor, labvitals_list_test, best_paths, k, test_point)
            print(best_distances[test_point][0][k])
        print()


if __name__ == "__main__":
    main()