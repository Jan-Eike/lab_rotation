import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from save_data import save_best_k, save_scores, save_temp_scores
from classification import classify, index_time_series_list


def find_best_k(labvitals_time_series_list_val, labels_val, k_list, save=True):
    """finds the best parameter k for knn.

    Args:
        labvitals_time_series_list_val (list of time Series): list of time series for validation set
        labels_val (List of integers): list of labels
        k_list (list of integers): List containing each value k that is going to be checked

    Returns:
        Integer: the parameter k that yielded the best result
    """
    # create dataset to save every intermediate result
    scores_dataframe = pd.DataFrame(columns=["k", "average acc score", "average auprc score"])
    avg_scores = []
    for k in tqdm(k_list, desc="finding best k"):
        avg_auprc_score, avg_roc_auc_score = cross_validate(labvitals_time_series_list_val, labels_val, k)
        # save scores in dictionary for each k
        scores_dict = {"k" : k, "average auprc score" : avg_auprc_score, "average roc auc score" : avg_roc_auc_score}
        scores_dataframe = scores_dataframe.append(scores_dict, ignore_index=True)
        avg_scores.append(avg_auprc_score)
        if save:
            save_temp_scores(scores_dict)
    if save:
        save_scores(scores_dataframe)
    max_index = np.argmax(avg_scores)
    if save:
        save_best_k(k_list[max_index])
    return k_list[max_index]


def cross_validate(labvitals_time_series_list_val, labels_val, k):
    """cross validation to compute an average score for the paramter k, using the validation set

    Args:
        labvitals_time_series_list_val (List of time Series): list of time series for validation set
        labels_val (List of integers): List of labels
        k (integer): parameter k that gets cross validated

    Returns:
            float: mean auprc_score of all scores achieved during cross validation
        float: mean roc_auc_score of all scores achieved during cross validation
    """
    splits = 3
    k_fold = KFold(n_splits=splits, random_state=42, shuffle=True)
    scores_auprc = []
    scores_roc_auc = []
    # just defining a progress bar for manual control
    pbar = tqdm(total=splits, desc="Cross validating for k = {}".format(k), leave=False)
    for train_index , test_index in k_fold.split(labvitals_time_series_list_val):
        x_train = index_time_series_list(labvitals_time_series_list_val, train_index)
        x_test = index_time_series_list(labvitals_time_series_list_val, test_index)
        y_train , y_test = labels_val[train_index] , labels_val[test_index]
        mean_score_auprc, mean_score_roc_auc = classify(x_train, x_test, y_train,
                                                        y_test, best_k=k, print_res=False)
        scores_auprc.append(mean_score_auprc)
        scores_roc_auc.append(mean_score_roc_auc)
        pbar.update(1)
    pbar.close()
    return np.mean(scores_auprc), np.mean(scores_roc_auc)
