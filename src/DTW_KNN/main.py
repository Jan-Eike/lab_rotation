import time
from classification import load_data, calculate_distance_matrices, classify, classify_precomputed
from find_hyperparameter import find_best_k
from save_data import load_best_k, load_distance_matrices
import click

@click.command()
@click.option('--use_saved_matrices', default=False, help='Want to use saved distance matrices from earlier calls?')
@click.option('--use_saved_k', default=False, help='Want to use saved best k from earlier calls?')
def main(use_saved_matrices=False, use_saved_k=False):
    """main method: Runs the entire clssification process."""
    # train, test and val length are just for testing purposes, to be able to
    # cut off parts of the datasets for faster computation
    train_length = 2500
    test_length = 625
    val_length = 250
    # list containing values for the hyperparameter k
    # these values are going to be used for finding the best k
    k_list = [1, 3, 5, 7, 9, 11, 13, 15, 33, 65, 129]

    start = time.time()

    data = load_data(train_length, test_length, val_length)
    labvitals_time_series_list_train, labels_train = data[0], data[3]
    labvitals_time_series_list_test, labels_test = data[1], data[4]
    labvitals_time_series_list_val, labels_val = data[2], data[5]

    best_k = get_best_k(use_saved_k, labvitals_time_series_list_val, labels_val, k_list)

    #dtw_matrices_train, dtw_matrices_test = get_distance_matrices(use_saved_matrices, labvitals_time_series_list_train,
    #                                                              labvitals_time_series_list_test, train_length, test_length)

    #classify_precomputed(dtw_matrices_train, dtw_matrices_test, labels_train, labels_test, test_length, best_k=best_k)

    classify(labvitals_time_series_list_train, labvitals_time_series_list_test, labels_train, labels_test, best_k=best_k, print_res=True)

    end = time.time()
    print("Time: {}".format(end - start))


def get_best_k(use_saved_k, labvitals_time_series_list_val, labels_val, k_list):
    """gets the best_k either from the database or from finding it 
       with the help of the method find_best_k

    Args:
        use_saved_k (Bool): should the value be taken from the databse?
        labvitals_time_series_list_val (List of time Series): Validation list containing time series for the labvitals
        labels_val (List of Integers): List of validation labels
        k_list (List of Integers): List of values for k to search for the best value for k

    Returns:
        Integer: the best k (k with highest score)
    """
    if use_saved_k:
        best_k = load_best_k()
    else:
        best_k = find_best_k(labvitals_time_series_list_val, labels_val, k_list)
    print("best k : {}".format(best_k))
    return best_k

# not used anymore
def get_distance_matrices(use_saved_train_matrices, labvitals_time_series_list_train, labvitals_time_series_list_test, train_length, test_length):
    """gets the distance matrices, either from the database or from calculating it

    Args:
        use_saved_train_matrices (Boolean): should the train matrices be taken from the database?
        labvitals_time_series_list_train (List of time Series): time series for training
        labvitals_time_series_list_test (List of time Series): time Series for testing
        train_length (int): lnegth of the training dataset (just for testing purposes)
        test_length (int): length of the test dataset (just for testing purposes)

    Returns:
        [type]: [description]
    """
    if use_saved_train_matrices:
        dtw_matrices_train = load_distance_matrices("dtw_matrices_train")
        dtw_matrices_test = calculate_distance_matrices(labvitals_time_series_list_train, labvitals_time_series_list_test,
                                                                    train_length, test_length, save=True, test_only=True)
    else:
        dtw_matrices_train, dtw_matrices_test = calculate_distance_matrices(labvitals_time_series_list_train, labvitals_time_series_list_test,
                                                                            train_length, test_length, save=True)
    return dtw_matrices_train, dtw_matrices_test


if __name__ == "__main__":
    main()
