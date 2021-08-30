import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from itertools import combinations


# method taken from dtaidistance.dtw_visualisation and modified
def plot_warping(s1, s2, path, distances, threshold=0):
    """Plot the optimal warping between to sequences.
    :param s1: From sequence.
    :param s2: To sequence.
    :param path: Optimal warping path.
    :param filename: Filename path (optional).
    """
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all')
    x1 = np.linspace(0, len(s1)-1, len(s1))
    x2 = np.linspace(0, len(s2)-1, len(s2))
    xy1 = np.vstack((x1,s1)).T
    xy2 = np.vstack((x2,s2)).T
    interval1, interval2 = check_slope(distances, threshold, path)
    mask1, mask2 = mask_interval(xy1, xy2, interval1, interval2)
    colors = {False : "blue", True : "red"}
    for i, (start, stop) in enumerate(zip(xy1[:-1], xy1[1:])):
            x, y = zip(start, stop)
            ax[0].plot(x, y, color=colors[mask1[i]])
    for i, (start, stop) in enumerate(zip(xy2[:-1], xy2[1:])):
            x, y = zip(start, stop)
            ax[1].plot(x, y, color=colors[mask2[i]])
    for int1, int2 in zip(interval1, interval2):
        for i in int1:
            ax[0].plot(xy1[i][0], xy1[i][1], "-o", color="red")
        for i in int2:
            ax[1].plot(xy2[i][0], xy2[i][1], "-o", color="red")
    plt.tight_layout()
    lines = []
    line_options = {'linewidth': 0.5, 'color': 'orange', 'alpha': 0.8}
    for r_c, c_c in path:
        if r_c < 0 or c_c < 0:
            continue
        con = ConnectionPatch(xyA=[r_c, s1[r_c]], coordsA=ax[0].transData,
                            xyB=[c_c, s2[c_c]], coordsB=ax[1].transData, **line_options)
        lines.append(con)
    for line in lines:
        fig.add_artist(line)
    return fig, ax


def check_slope(distances, threshold, path, length_threshold=2):
    """checks the slope of an imaginary function that takes
       the distances list as a function.
       It looks at all possible sublists of distances 
       and builds intervals in which the difference of distances
       is below the given threshold (this can be imagined as 
       the intervals of the corresponding imaginary function 
       the have a slope which is below the given threshold).

    Args:
        distances (list): List of sublists of distances between points in the optimal path
        threshold (float): maximum allowed distance
        path (list): list of tuples (points of the dtw matrix) which build a path
        length_threshold (int, optional): minimum length of sublists. Defaults to 2.

    Returns:
        tuple: two lists of intervals, each for its corresponding time series.
               Values in the subinterval mark similar positions.
    """
    # get all sublists of the distances list and their corresponding indices
    distances = [(distances[x:y], list(range(x,y))) for x, y in combinations(
            range(len(distances) + 1), r=2)]
    # only keep sublists of a certain length
    distances = [x for x in distances if len(x[0]) >= length_threshold]
    potential_intervals = create_potential_intervals(distances, threshold)
    merged_intervals = merge_common_sublists(potential_intervals)
    interval1, interval2 = transform_intervals(merged_intervals, path)
    return interval1, interval2


def create_potential_intervals(distances, threshold):
    """creates a list of intervals for which the difference in distances
       of the distances list is below a certain threshold. 

    Args:
        distances (list): List of sublists of distances between points in the optimal path
        threshold (float): maximum allowed distance

    Returns:
        list: potential intervals as described above.
    """
    potential_intervals = []
    for distance in distances:
        potential = True
        if len(distance[0]) != 0:
            for i in range(len(distance[0]) - 1):
                if abs(distance[0][i] - distance[0][i+1]) > threshold:
                    potential = False
            if potential == True:
                potential_intervals.append(distance)
    return potential_intervals


def merge_common_sublists(potential_intervals):
    """merges sublists of the given lists if one is a sublist of the other
       (wrt. the saved interval indices).

    Args:
        potential_intervals (list): List of intervals and their corresponding
                                    intervalsthat are going to be merged

    Returns:
        list: list of merged intervals (i.e. no interval in this list is a subset
              of another one, wrt. the saved interval indices).
    """
    # keeps track of deleted x and y values (to check if certain values
    # alredy got deleted).
    del_x = []
    del_y = []
    # we cant delete subintervals from the original list because
    # we would not be able to iterate properly over all subintervals
    new_potential_intervals = potential_intervals.copy()
    # go through every possible combination of two subintervals and check 
    # if one is a subset of another one.
    for x, y in combinations(range(len(potential_intervals)), r=2):
        if x not in del_x and y not in del_y and set(potential_intervals[x][1]).issubset(set(potential_intervals[y][1])):
            new_potential_intervals.remove(potential_intervals[x])
            del_x.append(x)
        if y not in del_y and x not in del_x and set(potential_intervals[y][1]).issubset(set(potential_intervals[x][1])):
            new_potential_intervals.remove(potential_intervals[y])
            del_y.append(y)
    return new_potential_intervals


def transform_intervals(merged_intervals, path):
    """transforms the given list of intervals, given the path, by
       changing the indices to be not an index of the distance list
       anymore, but of the corresponding time series.

    Args:
        merged_intervals (list): list of intervals 
        path (list): list of tuples (points of the dtw matrix)
                     which build a path

    Returns:
        tuple: two lists of intervals, each for its corresponding time series.
               Values in the subinterval mark similar positions.
    """
    interval1 = []
    interval2 = []
    for interval in merged_intervals:
        sub_interval1 = []
        sub_interval2 = []
        for i in interval[1]:
            sub_interval1.append(path[i][0])
            sub_interval2.append(path[i][1])
        # make a set to delete multiplicities
        interval1.append(list(set(sub_interval1)))
        interval2.append(list(set(sub_interval2)))
    return interval1, interval2


def mask_interval(xy1, xy2, interval1, interval2):
    """creates a mask for each time series that says which points
       have to marked as similar and which not.

    Args:
        xy1 (list): list of xy coordinates for the first time series.
        xy2 (list): list of xy coordinates for the second time series.
        interval1 (list): list of intervals that are marked as similar
                          for the first time series.
        interval2 (list): list of intervals that are marked as similar#
                          for the second time series.

    Returns:
        tuple: A tuple of two lists containing trues and falses to say which 
               points of each time series have to be marked as similar.
    """
    mask1 = []
    mask2 = []
    # init with False
    for i in range(len(xy1[1:])):
        mask1.append(False)
    for i in range(len(xy2[1:])):
        mask2.append(False)
    # go through each pair of intervals and mark them if two neighboring points
    # are in the same interval. 
    for interval_1, interval_2 in zip(interval1, interval2):
        for i, (start, stop) in enumerate(zip(xy1[:-1], xy1[1:])):
            if start[0] in interval_1 and stop[0] in interval_1:
                mask1[i] = True
        for i, (start, stop) in enumerate(zip(xy2[:-1], xy2[1:])):
            if start[0] in interval_2 and stop[0] in interval_2:
                mask2[i] = True
    return mask1, mask2