import json
import pickle
import gridfs
import pandas as pd
from pymongo import MongoClient
from bson.binary import Binary


def save_labvitals():
    """calls mongoimport for each labvitals dataset
    """
    mongoimport("labvitals_train", "../../data/train/full_labvitals.csv")
    mongoimport("labvitals_test", "../../data/test/full_labvitals.csv")
    mongoimport("labvitals_val", "../../data/val/full_labvitals.csv")


def mongoimport(coll_name, csv_path, db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the given csv file as coll_name to the mongo databse

    Args:
        coll_name (String): name of the collection
        csv_path (String): path to the csv file that is going to be saved
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the databse. Defaults to "mongodb://localhost:27017/".
    """
    data_base, collection = connect_to_database(coll_name, db_name=db_name, url=url)
    data_base.drop_collection(collection)
    data = pd.read_csv(csv_path)
    payload = json.loads(data.to_json(orient='records'))
    collection.insert_many(payload)


def save_scores(scores, db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the given score dataframe to the databse

    Args:
        scores (Pandas DataFrame): DataFrame of scores
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    data_base, collection_scores = connect_to_database("scores", db_name=db_name, url=url)
    collection_temp_scores = data_base["temp_scores"]
    data_base.drop_collection(collection_scores)
    payload = json.loads(scores.to_json(orient='records'))
    collection_scores.insert_many(payload)
    # temp scores get deleted after all scores have been computed
    data_base.drop_collection(collection_temp_scores)


def save_temp_scores(score, db_name="mongo", url="mongodb://localhost:27017/"):
    """saves scores for each k, until every score has been computed.

    Args:
        score (dictionary): dictionary containign the scores for a given k.
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    _, collection = connect_to_database("temp_scores", db_name=db_name, url=url)
    collection.insert_one(score)


def save_best_k(bets_result, collection_name="best_result",
                db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the best k (k that achieved the highest score).

    Args:
        bets_result (Int): the k that achieved the hightest score
        collection_name (str, optional): name of the collection. Defaults to "best_result".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    data_base, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    data_base.drop_collection(collection)
    collection.insert_one({"best_result" : bets_result})


def save_classification_data2(classification_data, best_paths, best_distances,
                             distances_per_test_point, dtw_matrices,
                             collection_name="classification_data",
                             db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the classification da as binary string

    Args:
        classification_data (List): List of Lists of the k nearest neighbors for each test datapoint
        best_paths (List): List of DTW Paths
        best_distances (List): List of best distances from DTW
        channel (Int): Channel number
        collection_name (str, optional): name of the collection. Defaults to "classification_data".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    _, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    serialized_classification_data = Binary(pickle.dumps(classification_data, protocol=2))
    best_paths = Binary(pickle.dumps(best_paths, protocol=2))
    best_distances = Binary(pickle.dumps(best_distances, protocol=2))
    distances_per_test_point = Binary(pickle.dumps(distances_per_test_point, protocol=2))
    dtw_matrices = Binary(pickle.dumps(dtw_matrices))
    neighbor_dict = {"nearest neighbors" : serialized_classification_data, "best paths": best_paths,
                     "best distances" : best_distances, "distances per test point": distances_per_test_point}
    collection.insert_one(neighbor_dict)
    db = MongoClient(url).mongo
    fs = gridfs.GridFS(db)
    fs.put(dtw_matrices)


def save_classification_data(classification_data, best_distances,
                             collection_name="classification_data",
                             db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the classification da as binary string

    Args:
        classification_data (List): List of Lists of the k nearest neighbors for each test datapoint
        best_paths (List): List of DTW Paths
        best_distances (List): List of best distances from DTW
        channel (Int): Channel number
        collection_name (str, optional): name of the collection. Defaults to "classification_data".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    _, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    serialized_classification_data = Binary(pickle.dumps(classification_data, protocol=2))
    best_distances = Binary(pickle.dumps(best_distances, protocol=2))
    neighbor_dict = {"nearest neighbors" : serialized_classification_data,
                     "best distances" : best_distances}
    collection.insert_one(neighbor_dict)


def delete_classification_data(collection_name="classification_data",
                               collection_name2="nn with false label",
                               collection_name3="fs.chunks",
                               collection_name4="fs.files",
                               db_name="mongo", url="mongodb://localhost:27017/"):
    """deletes the classification collection.
       this should be done before starting to fill it up again
       this is not done in the save method because
       it will be called multiple times in a row

    Args:
        collection_name (str, optional): name of the collection. Defaults to "classification_data".
        collection_name2 (str, optional): name of the  second collection. Defaults to "nn with false label".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    database, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    database.drop_collection(collection)
    database2, collection2 = connect_to_database(collection_name2)
    database2.drop_collection(collection2)
    database3, collection3 = connect_to_database(collection_name3)
    database3.drop_collection(collection3)
    database4, collection4 = connect_to_database(collection_name4)
    database4.drop_collection(collection4)


def save_current_test_data(test_data, collection_name="current_test_data",
                           db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the currently used test data as binary string. This will
       be used when we don't use the entire test dataset

    Args:
        test_data (List of DataFrames): curent test data as described above
        collection_name (str, optional): name of the collection. Defaults to "current_test_data".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    database, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    database.drop_collection(collection)
    collection.insert_one({"current test data" : Binary(pickle.dumps(test_data, protocol=2))})


def save_nn_with_false_label(nn_with_false_label, true_label, false_label, collection_name="nn with false label"):
    """saves the nearest neighbor with a different label as the test point

    Args:
        nn_with_false_label (DataFrame): the nearest neighbor DataFrame
        true_label (int): predicted label
        false_label (int): label of nn_with_false_label
        collection_name (str, optional): name of the collection. Defaults to "nn with false label".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    _, collection = connect_to_database(collection_name)
    collection.insert_one({"nn" : Binary(pickle.dumps(nn_with_false_label)), 
                           "true_label" : Binary(pickle.dumps(true_label)),
                           "false_label" : Binary(pickle.dumps(false_label))})


def save_predicted_labels(pred_labels, collection_name="predicted_labels"):
    """saves predicted labels

    Args:
        pred_labels (list): list of predicted labels
        collection_name (str, optional): name of the collection. Defaults to "predicted_labels".
    """
    _, collection = connect_to_database(collection_name)
    collection.insert_one({"pred_labels" : Binary(pickle.dumps(pred_labels))})


def delete_predicted_labels(collection_name="predicted_labels"):
    """deletes the predicted labels collection

    Args:
        collection_name (str, optional): name of the collection. Defaults to "predicted_labels".
    """
    database, collection = connect_to_database(collection_name)
    database.drop_collection(collection)



def save_test_labels(test_labels, collection_name="test_labels"):
    """saves test labels

    Args:
        pred_labels (list): list of test labels
        collection_name (str, optional): name of the collection. Defaults to "test_labels".
    """
    _, collection = connect_to_database(collection_name)
    collection.insert_one({"test_labels" : Binary(pickle.dumps(test_labels))})


def delete_test_labels(collection_name="test_labels"):
    """deletes the test labels collection

    Args:
        collection_name (str, optional): name of the collection. Defaults to "test_labels".
    """
    database, collection = connect_to_database(collection_name)
    database.drop_collection(collection)


def connect_to_database(collection_name, db_name="mongo", url="mongodb://localhost:27017/"):
    """connects to certain collection in database

    Args:
        collection_name (String): name of the collection
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".

    Returns:
        database
        collection
    """
    client = MongoClient(url)
    database = client[db_name]
    collection = database[collection_name]
    return database, collection


if __name__ == "__main__":
    delete_predicted_labels()
    delete_test_labels()
    delete_classification_data()
