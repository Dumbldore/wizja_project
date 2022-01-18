import numpy as np
import matplotlib.pyplot as plt
from classify_trajectories import extract_norm
from load_trackers import load_trackers

trackers = load_trackers(path="../new_coco_kalman_highway.mp4.json")
#trackers = load_trackers(path="../new_coco_kalman_highway.mp4.json")
norms = extract_norm(trackers, epsilon=100, min_samples=3)
np.save("norms.npy", norms)

fig = plt.figure()
ax = fig.add_subplot(111)
for traj in norms:

    ax.plot(traj[:, 0], traj[:, 1], c="k", linestyle="dashed")
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.gca().invert_xaxis()
plt.show()
