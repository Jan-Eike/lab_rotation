import time
import click
import multiprocessing
from classification import classify
from find_hyperparameter import find_best_k
from load_data import load_best_k
from save_data import save_test_labels
from data_loader import load_data


@click.command()
@click.option('--use_saved_k', default=False,
              help='Want to use saved best k from earlier calls?')
@click.option('--train_length', default=-1,
              help='Length of the train dataset. Use entire train set with -1 (default).')
@click.option('--test_length_start', default=0,
              help='Length of the test dataset. Use entire test set with -1 (default).')
@click.option('--test_length_end', default=-1,
              help='Length of the test dataset. Use entire test set with -1 (default).')
@click.option('--val_length', default=-1,
              help='Length of the validation dataset. Use entire validation set with -1 (default).')
@click.option('--save_classification', default=False,
              help='Want to save the classification results?')
@click.option('--result', default=True,
              help='Want to calculate the results?')
@click.option('--num_cores', default=-1,
              help='Number of cores for multitasking. Use maximum number of cores - 2 with -1 (default)')
def main(use_saved_k=False, train_length=-1, val_length=-1, test_length_start=-1, test_length_end=-1, save_classification=False, result=True, num_cores=-1):
    """main method: Runs the entire clssification process."""
    train_length = int(train_length)
    test_length_start = int(test_length_start)
    test_length_end = int(test_length_end)
    val_length = int(val_length)

    # list containing values for the hyperparameter ks
    # these values are going to be used for finding the best k
    k_list = [1, 3, 5, 7, 9, 11, 13, 15, 17]

    start = time.time()

    data = load_data(train_length, test_length_start, test_length_end, val_length)
    labvitals_time_series_list_train, labels_train = data[0], data[3]
    labvitals_time_series_list_test, labels_test = data[1], data[4]
    labvitals_time_series_list_val, labels_val = data[2], data[5]
    train_length = len(labvitals_time_series_list_train)
    test_length = len(labvitals_time_series_list_test)
    val_length = len(labvitals_time_series_list_val)

    save_test_labels((labels_test, test_length_start//test_length))

    best_k = get_best_k(use_saved_k, labvitals_time_series_list_val, labels_val, k_list)
    best_k = 1
    classify(labvitals_time_series_list_train, labvitals_time_series_list_test,
             labels_train, labels_test, test_length_start, test_length, best_k=best_k, print_res=True,
             save_classification=save_classification, num_cores=num_cores, result=result)
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


if __name__ == "__main__":
    main()
