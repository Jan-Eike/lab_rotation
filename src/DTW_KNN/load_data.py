import pickle
from save_data import connect_to_database

def load_nn_with_false_label(collection_name="nn with false label",
                             db_name="mongo", url="mongodb://localhost:27017/"):
    """loads the nearest neighbor with a different label as the test point

    Args:
        collection_name (str, optional): name of the collection. Defaults to "nn with false label".
        db_name (str, optional): Name of the database. Defaults to "mongo".
        url (str, optional): url to the database. Defaults to "mongodb://localhost:27017/".

    Returns:
        DataFrame: the nearest neighbor with a different label as the test point
    """
    _, collection = connect_to_database(collection_name, db_name=db_name, url=url)
    cursor = collection.find({})
    nearest_neighbor = []
    for data in cursor:
        nearest_neighbor.append(pickle.loads(data["nn"]))
    return nearest_neighbor


def load_current_test_data(collection_name="current_test_data",
                           db_name="mongo", url="mongodb://localhost:27017/"):
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


def load_classification_data(collection_name="classification_data",
                             db_name="mongo", url="mongodb://localhost:27017/"):
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
