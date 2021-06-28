from save_data import load_classification_data, load_current_test_data
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_visualisation as dtwvis
from tslearn import metrics


def main():
    """Main method for explainabilty.
    """
    k_nearest_time_series, best_paths, best_distances, distances_per_test_point  = load_classification_data()
    print(best_distances)
    #print(k_nearest_time_series[0][0])
    print(len(best_paths))
    print(best_paths)
    labvitals_list_test = load_current_test_data()
    #print(labvitals_list_test[0])
    #plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, channel[1], 0)
    print(labvitals_list_test[0])
    print(np.array(labvitals_list_test[0].iloc[:, 6:].iloc[:, [0]], dtype='float64').reshape(-1,))
    #print(channel)
    plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, distances_per_test_point)
    print(best_distances[0][0])
    print(distances_per_test_point[0][0][0])


def plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, nn, test_point, distances_per_test_point, distance):
    number_of_channels = labvitals_list_test[test_point].iloc[:, 6:].shape[1]
    for channel in range(number_of_channels):
        # the first indexing number stands for the current test point when this will be done automatically in the end (or not idk)

        # labvitals_list_test[test_point] takes the test_pointsth DataFrame from the list
        # iloc[:, 6:] cuts off the first 5 unnecessary columns
        # iloc[:, [channel]] takes the column with the index channel
        # np.array transforms it to a numpy array
        # reshape(-1,) reshapes it from (a,1) to (a,) (from 2 dimensions to 1)
        time_series_1 = np.array(k_nearest_time_series[test_point][nn].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
        time_series_2 = np.array(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
        print(k_nearest_time_series[test_point][0].iloc[:, 6:].iloc[:, [channel]])
        print(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]])
        fig, ax = dtwvis.plot_warping(time_series_2, time_series_1, best_paths[test_point][0])
        fig.set_size_inches(15, 8)
        fig.subplots_adjust(hspace=0.2)
        fig.suptitle(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]].columns[0], y=1)
        ax[0].set_title("test data time series", y=1, pad=-14)
        ax[1].set_title("nearest neighbor", y=1, pad=-14)
        d = metrics.dtw(time_series_2, time_series_1)
        print(d)
        print(distances_per_test_point[0][test_point][nn][channel])
        #plt.show()
        distance += d
    return distance


def plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, distances_per_test_point):
    nn = 0
    test_point = 0
    distance = plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, nn, test_point, distances_per_test_point, 0)
    print(distance / 24)


if __name__ == "__main__":
    main()