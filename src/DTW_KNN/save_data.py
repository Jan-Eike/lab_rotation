import pandas as pd
from pymongo import MongoClient
from bson.binary import Binary
import json
import pickle


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
    db, collection = connect_to_database(coll_name, db_name=db_name, url=url)
    db.drop_collection(collection)
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
    db, collection_scores = connect_to_database("scores", db_name=db_name, url=url)
    collection_temp_scores = db["temp_scores"]
    db.drop_collection(collection_scores)
    payload = json.loads(scores.to_json(orient='records'))
    collection_scores.insert_many(payload)
    # temp scores get deleted after all scores have been computed
    db.drop_collection(collection_temp_scores)


def save_temp_scores(score, db_name="mongo", url="mongodb://localhost:27017/"):
    """saves scores for each k, until every score has been computed.

    Args:
        score (dictionary): dictionary containign the scores for a given k.
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    _, collection = connect_to_database("temp_scores", db_name=db_name, url=url)
    collection.insert_one(score)
    

def save_distance_matrices(matrices, collection_name, db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the distance matrices.

    Args:
        matrices (List of Numpy ndarrays): List of distance matrices
        collection_name (String): Name of the collection
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    db, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    db.drop_collection(collection)
    for matrix in matrices:
        serialized_matrix = Binary(pickle.dumps(matrix, protocol=2))
        serialized_matrix_dict = {'{}_distance_matrix'.format(collection_name) : serialized_matrix}
        collection.insert_one(serialized_matrix_dict)

def load_distance_matrices(collection_name, db_name="mongo", url="mongodb://localhost:27017/"):
    """loads matrix from the distance_matrices collection and transforms it to a numpy ndarray

    Args:
        collection_name (String) : Name of the collection.
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".

    Returns:
        List of numpy ndarrays: list of distance matrices
    """
    _, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    cursor = collection.find({})
    matrices = []
    for matrix in cursor:
        matrices.append(pickle.loads(matrix['{}_distance_matrix'.format(collection_name)]))
    return matrices


def save_best_k(bets_result, collection_name="best_result", db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the best k (k that achieved the highest score).

    Args:
        bets_result (Int): the k that achieved the hightest score
        collection_name (str, optional): name of the collection. Defaults to "best_result".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    db, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    db.drop_collection(collection)
    collection.insert_one({"best_result" : bets_result})


def load_best_k(collection_name="best_result", db_name="mongo", url="mongodb://localhost:27017/"):
    """loads the best k

    Args:
        collection_name (str, optional): name of the collection. Defaults to "best_result".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".

    Returns:
        Int: best k
    """
    _, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    cursor = collection.find({})
    best_k = []
    for k in cursor:
        best_k.append(k["best_result"])
    return best_k[0]


def save_classification_data(classification_data, best_paths, best_distances, distances_per_test_point, collection_name="classification_data", db_name="mongo", url="mongodb://localhost:27017/"):
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
    neighbor_dict = {"nearest neighbors" : serialized_classification_data, "best paths": best_paths,
                     "best distances" : best_distances, "distances per test point": distances_per_test_point}
    collection.insert_one(neighbor_dict)


def delete_classification_data(collection_name="classification_data", collection_name2="nn with false label", db_name="mongo", url="mongodb://localhost:27017/"):
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
    db, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    db.drop_collection(collection)
    db2, collection2 = connect_to_database(collection_name2)
    db2.drop_collection(collection2)


def load_classification_data(collection_name="classification_data", db_name="mongo", url="mongodb://localhost:27017/"):
    """loads the classification data and transforms it back to normal

    Args:
        collection_name (str, optional): name of the collection. Defaults to "classification_data".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".

    Returns:
        Lists: All the classification results, the best paths, the best distances, all distances per test point
    """
    _, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    cursor = collection.find({})
    classification_data = []
    best_distances = []
    best_paths = []
    distances_per_test_point = []
    for data in cursor:
        classification_data.append(pickle.loads(data["nearest neighbors"]))
        best_paths.append(pickle.loads(data["best paths"]))
        best_distances.append(pickle.loads(data["best distances"]))
        distances_per_test_point.append(pickle.loads(data["distances per test point"]))
    # last entry, because the elements get added to a list but the save operation is called after every 
    # append, so only after the last append everything is saved.
    return classification_data, best_paths, best_distances, distances_per_test_point


def save_current_test_data(test_data, collection_name="current_test_data", db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the currently used test data as binary string. This will 
       be used when we don't use the entire test dataset

    Args:
        test_data (List of DataFrames): curent test data as described above
        collection_name (str, optional): name of the collection. Defaults to "current_test_data".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    db, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    db.drop_collection(collection)
    collection.insert_one({"current test data" : Binary(pickle.dumps(test_data, protocol=2))})


def load_current_test_data(collection_name="current_test_data", db_name="mongo", url="mongodb://localhost:27017/"):
    """loads the current test data and converts it back to normal

    Args:
        collection_name (str, optional): name of the collection. Defaults to "current_test_data".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".

    Returns:
        List of DataFrames: List of the current test DataFrames
    """
    _, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    cursor = collection.find({})
    test_data = []
    for data in cursor:
        test_data.append(pickle.loads(data["current test data"]))
    return test_data[0]


def save_nn_with_false_label(nn_with_false_label, collection_name="nn with false label", db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the nearest neighbor with a different label as the test point

    Args:
        nn_with_false_label (DataFrame): teh nearest neighbor DataFrame
        collection_name (str, optional): name of the collection. Defaults to "nn with false label".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    db, collection = connect_to_database(collection_name)
    collection.insert_one({"nn" : Binary(pickle.dumps(nn_with_false_label))})


def load_nn_with_false_label(collection_name="nn with false label", db_name="mongo", url="mongodb://localhost:27017/"):
    """loads the nearest neighbor with a different label as the test point

    Args:
        collection_name (str, optional): name of the collection. Defaults to "nn with false label".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".

    Returns:
        DataFrame: the nearest neighbor with a different label as the test point
    """
    _, collection = connect_to_database(collection_name)
    cursor = collection.find({})
    nn = []
    for data in cursor:
        nn.append(pickle.loads(data["nn"]))
    return nn


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
    db = client[db_name]
    collection = db[collection_name]
    return db, collection
