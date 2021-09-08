import pickle
import gridfs
from pymongo import MongoClient
from save_data import connect_to_database

def load_nn_with_false_label(collection_name="nn with false label"):
    """loads the nearest neighbor with a different label as the test point

    Args:
        collection_name (str, optional): name of the collection. Defaults to "nn with false label".

    Returns:
        DataFrame: the nearest neighbor with a different label as the test point
    """
    _, collection = connect_to_database(collection_name)
    cursor = collection.find({})
    nearest_neighbor = []
    true_label = []
    false_label = []
    for data in cursor:
        nearest_neighbor.append(pickle.loads(data["nn"]))
        true_label.append(pickle.loads(data["true_label"]))
        false_label.append(pickle.loads(data["false_label"]))
    return nearest_neighbor, true_label, false_label


def load_current_test_data(collection_name="current_test_data"):
    """loads the current test data and converts it back to normal

    Args:
        collection_name (str, optional): name of the collection. Defaults to "current_test_data".

    Returns:
        List of DataFrames: List of the current test DataFrames
    """
    _, collection = connect_to_database(collection_name)
    cursor = collection.find({})
    test_data = []
    for data in cursor:
        test_data.append(pickle.loads(data["current test data"]))
    return test_data[0]


def load_classification_data(collection_name="classification_data"):
    """loads the classification data and transforms it back to normal

    Args:
        collection_name (str, optional): name of the collection. Defaults to "classification_data".

    Returns:
        Lists: All the classification results, the best paths, the best distances, all distances per test point
    """
    _, collection = connect_to_database(collection_name)
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
    return classification_data, best_paths, best_distances, distances_per_test_point



def load_best_k(collection_name="best_result"):
    """loads the best k

    Args:
        collection_name (str, optional): name of the collection. Defaults to "best_result".

    Returns:
        Int: best k
    """
    _, collection = connect_to_database(collection_name)
    cursor = collection.find({})
    best_k = []
    for k in cursor:
        best_k.append(k["best_result"])
    return best_k[0]


def load_predicted_labels(collection_name="predicted_labels"):
    _, collection = connect_to_database(collection_name)
    cursor = collection.find({})
    pred_labels = []
    for pred_label in cursor:
        pred_labels.append(pickle.loads(pred_label["pred_labels"]))
    return pred_labels


def load_test_labels(collection_name="test_labels"):
    _, collection = connect_to_database(collection_name)
    cursor = collection.find({})
    test_labels = []
    for test_label in cursor:
        test_labels.append(pickle.loads(test_label["test_labels"]))
    return test_labels


def load_dtw_matrices(collection_name1="fs.files", collection_name2="fs.chunks",
                      url="mongodb://localhost:27017/"):
    _, collection1 = connect_to_database(collection_name1)
    cursor = collection1.find({})
    ids = []
    for file in cursor:
        ids.append(file["_id"])
    
    db = MongoClient(url).mongo
    fs = gridfs.GridFS(db)
    dtw_matrices = []
    for id in ids:
        dtw_matrices.append(pickle.loads(fs.get(id).read()))
    return dtw_matrices