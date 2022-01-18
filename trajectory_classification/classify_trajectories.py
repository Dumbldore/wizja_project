import math

import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.optimize import curve_fit
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
from util import func_classify


def hausdorff(u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def extract_norm(trackers, epsilon, min_samples):
    locations = []
    traj_lst = []
    for tracker in trackers:
        x = [int(track.x) for track in tracker.tracks]
        y = [int(track.y) for track in tracker.tracks]
        z = [int(track.frame) for track in tracker.tracks]
        if len(x) < 40:
            continue
        if max(x) - min(x) < 150 and max(y) - min(y) < 150:
            continue
        z = [el - min(z) for el in z]
        try:
            poptx, pcovx = curve_fit(func_classify, z, x, maxfev=100000)
            popty, pcovy = curve_fit(func_classify, z, y, maxfev=100000)
        except:
            continue
        zmin = min(z)
        zmax = max(z)
        zdata = np.linspace(zmin, zmax, (zmax - zmin))
        xdata = [func_classify(zd, *poptx) for zd in zdata]
        ydata = [func_classify(zd, *popty) for zd in zdata]
        if max(xdata) - min(xdata) < 450 and max(ydata) - min(ydata) < 150:
            continue
        locations.append(np.array([x, y]).T)
        traj = np.array([xdata, ydata]).T
        traj_lst.append(traj)

    traj_lst = np.array(traj_lst)

    degree_threshold = 5

    for traj_index, traj in enumerate(traj_lst):
        hold_index_lst = []
        previous_azimuth = 1000
        for point_index, point in enumerate(traj[:-1]):
            next_point = traj[point_index + 1]
            diff_vector = next_point - point
            azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360

            if abs(azimuth - previous_azimuth) > degree_threshold:
                hold_index_lst.append(point_index)
                previous_azimuth = azimuth
        hold_index_lst.append(
            traj.shape[0] - 1
        )  # Last point of trajectory is always added

        traj_lst[traj_index] = traj[hold_index_lst, :]

    traj_count = len(traj_lst)
    D = np.zeros((traj_count, traj_count))

    for i in range(traj_count):
        for j in range(i + 1, traj_count):
            distance = hausdorff(traj_lst[i], traj_lst[j])
            D[i, j] = distance
            D[j, i] = distance
    mdl = DBSCAN(eps=epsilon, min_samples=min_samples)
    cluster_lst = mdl.fit_predict(D)
    trajectories = np.array(locations)
    color_lst = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for traj, cluster in zip(traj_lst, cluster_lst):

        if cluster == -1:
            ax.plot(traj[:, 0], traj[:, 1], c="k", linestyle="dashed")

        else:
            ax.plot(traj[:, 0], traj[:, 1], c=color_lst[cluster % len(color_lst)])
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.gca().invert_xaxis()
    plt.show()

    sum = 0
    number = 0
    minimal = np.inf
    chosen = [0] * (max(cluster_lst) + 1)
    for j in range((max(cluster_lst) + 1)):
        for i, trajectory in enumerate(trajectories):
            if cluster_lst[i] == j:
                for k, traj in enumerate(trajectories):
                    if cluster_lst[k] == j:
                        sum += fastdtw(trajectory, traj)[0]
                        number += 1
                result = sum / number
                if result < minimal:
                    minimal = result
                    chosen[j] = trajectory
                sum = 0
                number = 0
                minimal = np.inf
    return chosen
