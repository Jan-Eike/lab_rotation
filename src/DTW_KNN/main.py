import time
from dynamic_time_warping import load_data, calculate_distance_matrices, classify, find_best_k
from save_data import load_best_k, load_distance_matrices
from sklearn.model_selection import KFold 

def main(use_saved_matrices=False, use_saved_k=False):
    """main method: Runs the entire clssification process."""
    # train, test and val length are just for testing purposes, to be able to
    # cut off parts of the datasets for faster computation
    train_length = 300
    test_length = 66
    val_length = 200
    start = time.time()
    data = load_data(train_length, test_length, val_length)
    labvitals_time_series_list_train, labels_train = data[0], data[3]
    labvitals_time_series_list_test, labels_test = data[1], data[4]
    labvitals_time_series_list_val, labels_val = data[2], data[5]
    k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if use_saved_k:
        best_k = load_best_k()
    else:
        best_k = find_best_k(labvitals_time_series_list_val, labels_val, k_list)
    print("best k : {}".format(best_k))
    if use_saved_matrices:
        dtw_matrices_train = load_distance_matrices("dtw_matrices_train")
        dtw_matrices_test = load_distance_matrices("dtw_matrices_test")
        print(dtw_matrices_train)
    else:
        dtw_matrices_train, dtw_matrices_test = calculate_distance_matrices(labvitals_time_series_list_train, labvitals_time_series_list_test, train_length, test_length, save=True)
    classify(dtw_matrices_train, dtw_matrices_test, labels_train, labels_test, test_length, best_k=best_k)
    end = time.time()
    print("Time: {}".format(end - start))






if __name__ == "__main__":
    main(use_saved_matrices=False, use_saved_k=False)
    main(use_saved_matrices=True, use_saved_k=True)