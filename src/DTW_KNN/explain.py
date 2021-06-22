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


def plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, channel):
    # plot for test point 0
    time_series_1 = np.array(k_nearest_time_series[0][0].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
    time_series_2 = np.array(labvitals_list_test[0].iloc[:, 6:].iloc[:, [channel]], dtype='float64').reshape(-1,)
    # the first 0 stands for the current test point when this will be done automatically in the end
    print(k_nearest_time_series[0][0].iloc[:, 6:].iloc[:, [channel]])
    print(labvitals_list_test[0].iloc[:, 6:].iloc[:, [channel]])
    dtwvis.plot_warping(time_series_2, time_series_1, best_paths[0][0])
    plt.show()


def plot_all_channels(k_nearest_time_series, labvitals_list_test, best_paths, channels):
    for channel in channels:
        plot_explain(k_nearest_time_series, labvitals_list_test, best_paths, channel)


if __name__ == "__main__":
    main()