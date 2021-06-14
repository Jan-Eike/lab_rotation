import pandas as pd
from pymongo import MongoClient
from bson.binary import Binary
import json
import pickle
import numpy as np


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
    client = MongoClient(url)
    db = client[db_name]
    collection = db[coll_name]
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
    client = MongoClient(url)
    db = client[db_name]
    collection = db["scores"]
    db.drop_collection(collection)
    payload = json.loads(scores.to_json(orient='records'))
    collection.insert_many(payload)


def save_distance_matrix(matrix, db_name="mongo", url="mongodb://localhost:27017/"):
    """saves the given matrix to the database

    Args:
        matrix (numpy ndarray): distance matrix
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".
    """
    client = MongoClient(url)
    db = client[db_name]
    collection = db["distance_matrices"]
    # serialize matrix with pickle to save it in the database
    serialized_matrix = Binary(pickle.dumps(matrix, protocol=2))
    json.loads(serialized_matrix)
    collection.insert_one(serialized_matrix)

if __name__ == "__main__":
    matrix = np.array(
        [[0, 5, 9],
         [5, 0, 7],
         [9, 7, 0]]
    )
    save_distance_matrix(matrix)