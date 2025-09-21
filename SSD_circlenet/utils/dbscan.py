import numpy as np


def euclidean(x, y):
    """
    :param x: np.array of coords of point x
    :param y: np.array of coords of point y
    :return: euclidean dist between point x and y
    """
    return np.linalg.norm(x - y)


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=1, metric=euclidean):
        """
        The cluster labels are set in self._labels after DBSCAN.fit(X) is called
        Noise points are labeled as -1

        If min_samples is set to 1, no points are noise points

        :param eps: max dist between points to be considered a cluster
        :param min_samples: min samples for a cluster to form or point is noise
        :param metric: distance metric function.
            Must take two points as input and return distance between them
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self._labels = None

    def fit(self, X) -> None:
        """
        :param X: np.array of points in space to be clustered
        :return: None
        """
        if not (isinstance(X, np.ndarray)):
            raise ValueError(f"The value X must be numpy array. {type(X)} used instead")

        # pts labeled as 0 are unassigned
        self._labels = [0] * len(X)
        # id of current cluster
        curr_id = 0

        # check if points are candidate for core/seed points
        for i in range(len(X)):
            # only consider unlabeled points
            if self._labels[i] == 0:
                neigh_pts_idx = self._get_neigh_pts_idx(i, X)

                # core points with less than min_samples neigh pts are noise
                # noise can be relabeled to other pts later
                if len(neigh_pts_idx) < self.min_samples:
                    self._labels[i] = -1
                else:
                    curr_id += 1
                    self._grow_cluster(i, curr_id, neigh_pts_idx, X)

    def _get_neigh_pts_idx(self, idx, X):
        """
        Gets all the idxs of points that are within eps distance of point with index idx
        :param idx:
        :return: np.array of neighbouring points
        """
        return np.array(
            [i for i, x in enumerate(X) if self.metric(x, X[idx]) <= self.eps]
        )

    def _grow_cluster(self, idx, curr_id, neigh_pts_idx, X):
        self._labels[idx] = curr_id

        i = 0
        while i < len(neigh_pts_idx):
            neigh_idx = neigh_pts_idx[i]

            if self._labels[neigh_idx] == -1:
                # if the neigh ptn is noise, it becomes part of seed ptn curr_id
                self._labels[neigh_idx] = curr_id
            elif self._labels[neigh_idx] == 0:
                # if the neigh ptn is unassigned
                self._labels[neigh_idx] = curr_id

                cur_neigh_ptn_idx = self._get_neigh_pts_idx(neigh_idx, X)
                # If neigh_idx has self.min_samples neighbors, add it as branch point
                if len(cur_neigh_ptn_idx) >= self.min_samples:
                    neigh_pts_idx = np.append(neigh_pts_idx, cur_neigh_ptn_idx, axis=0)
            i += 1
