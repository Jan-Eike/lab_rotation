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
