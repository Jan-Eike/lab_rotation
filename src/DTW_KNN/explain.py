from save_data import load_classification_data, load_current_test_data
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_visualisation as dtwvis


def main():
    """Main method for explainabilty.
    """
    channel, k_nearest_time_series, best_paths  = load_classification_data()
    labvitals_list_test = load_current_test_data()
    plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, channel[1])
    plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, channel)


def plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, channel):
    # the first indexing number stands for the current test point when this will be done automatically in the end (or not idk)
    test_point = 2
    time_series_1 = np.array(k_nearest_time_series[test_point][0].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
    time_series_2 = np.array(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
    print(k_nearest_time_series[test_point][0].iloc[:, 6:].iloc[:, [channel]])
    print(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]])
    fig, ax = dtwvis.plot_warping(time_series_2, time_series_1, best_paths[2][0])
    fig.set_size_inches(15, 8)
    fig.subplots_adjust(hspace=0.2)
    fig.suptitle(labvitals_list_test[test_point].iloc[:, 6:].iloc[:, [channel]].columns[0], y=1)
    ax[0].set_title("test data time series", y=1, pad=-14)
    ax[1].set_title("nearest neighbor", y=1, pad=-14)
    plt.show()


def plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, channels):
    for channel in channels:
        plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, channel)


if __name__ == "__main__":
    main()