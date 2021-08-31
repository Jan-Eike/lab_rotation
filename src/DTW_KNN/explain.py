import statistics
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from explainability_utils import plot_warping
from dtw_utils import dtw_path
from load_data import (load_classification_data,
                       load_current_test_data, 
                       load_nn_with_false_label)


def main():
    """Main method for explainabilty.
    """
    k_nearest_time_series, best_paths, best_distances, distances_per_test_point, dtw_matrices  = load_classification_data()
    labvitals_list_test = load_current_test_data()
    nn_with_false_label, true_label, false_label = load_nn_with_false_label()
    # re-sort everything because parallelization mixes up the order randomly
    k_nearest_time_series = sorted(k_nearest_time_series, key=itemgetter(1))
    best_paths = sorted(best_paths, key=itemgetter(1))
    best_distances = sorted(best_distances, key=itemgetter(1))
    dtw_matrices = sorted(dtw_matrices, key=itemgetter(1))
    distances_per_test_point = sorted(distances_per_test_point, key=itemgetter(1))
    plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, best_distances,
                      nn_with_false_label, false_label, dtw_matrices, distances_per_test_point)


def plot_explain(nearest_neighbor, labvitals_list_test, best_paths, nn, test_point,
                 nn_with_false_label, false_label, dtw_matrices, distances_for_test_point):
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
    print(distances_for_test_point)
    sorted_indices = np.argsort(distances_for_test_point)
    for channel in sorted_indices[:5]:
        # the first indexing number stands for the current test point when this will be done automatically in the end (or not idk)

        # labvitals_list_test[test_point] takes the test_point-th DataFrame from the list
        # iloc[:, 6:] cuts off the first 5 unnecessary columns
        # iloc[:, [channel]] takes the column with the index channel
        # np.array transforms it to a numpy array
        # reshape(-1,) reshapes it from (a,1) to (a,) (from 2 dimensions to 1)
        time_series_1 = np.array(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
        time_series_2 = np.array(nearest_neighbor.iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
        time_series_nn_with_false_label = np.array(nn_with_false_label.iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
        
        if plot_dtw:
            print(best_paths[channel])
            print(len(time_series_1), len(time_series_2))
            best_path = best_paths[channel]
            print(distances_for_test_point)
            distances = []
            for point in best_path:
                distances.append(dtw_matrices[channel][point])
            print(["{:.2f}".format(i) for i in distances])
            print(distances)
            #index_largest_gap = np.argsort(np.diff(lam))[::-1][:-1]
            differences = sorted([j-i for i, j in zip(distances[:-1], distances[1:])])
            print(differences)
            distance_mean = np.mean(differences)
            #distance_median = statistics.median(differences)
            fig, ax = plot_warping(time_series_1, time_series_2, best_path, distances, threshold=distance_mean)
            fig.set_size_inches(9, 5)
            fig.subplots_adjust(hspace=0.2)
            fig.suptitle("{}, {}".format(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]].columns[0], test_point), y=1)
            ax[0].set_title("test data time series", y=1, pad=-14)
            ax[1].set_title("nearest neighbor, {}".format(false_label), y=1, pad=-14)

            """
            path, d, matrix = dtw_path(time_series_1, time_series_nn_with_false_label)
            fig2, ax2 = plot_warping(time_series_1, time_series_nn_with_false_label, path)
            fig2.set_size_inches(9, 5)
            fig2.subplots_adjust(hspace=0.2)
            fig2.suptitle(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]].columns[0], y=1)
            ax2[0].set_title("test data time series", y=1, pad=-14)
            ax2[1].set_title("nearest neighbor with false label, {}".format(1 - false_label), y=1, pad=-14)
            """
            plt.show()


def plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, best_distances,
                      nn_with_false_label, false_label, dtw_matrices, distances_per_test_point):
    """calls plot_explain for each test point in the test_point list
       and prints out the best distance from each test point to its k nearest neighbors.

    Args:
        k_nearest_time_series (List): List of the DataFrames of the k nearest neighbors
        labvitals_list_test (List of DataFrames): List of the test DataFrames
        best_paths (List): List of the best dtw paths for each test point
        best_distances (List): List of the smallest distances for each Test point
    """
    test_points = [0,1,2,3,4,5,6,7,8,9]
    for test_point in test_points:
        print("Test point {}:".format(test_point))
        for k, nearest_neighbor in enumerate(k_nearest_time_series[test_point][0]):
            plot_explain(nearest_neighbor, labvitals_list_test, best_paths[test_point][0][k], k, test_point,
                         nn_with_false_label[test_point], false_label[test_point],
                         dtw_matrices[test_point][0][k], distances_per_test_point[test_point][0][k])
            print(best_distances[test_point][0][k])
        print()


if __name__ == "__main__":
    main()