# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, max_frame_tracking=50):
        """
        :param max_frame_tracking: max number of frames to wait
        before removing previously tracked that disappeared
        """
        self.nextObjectID = 1
        self.obj_bbox = OrderedDict()
        self.obj_centroids = OrderedDict()
        self.obj_absence_counts = OrderedDict()
        self.obj_distance_counts = OrderedDict()
        self.max_frame_tracking = max_frame_tracking

    def register(self, centroid, bbox=None):
        self.obj_centroids[self.nextObjectID] = centroid
        self.obj_absence_counts[self.nextObjectID] = 0
        self.obj_distance_counts[self.nextObjectID] = []
        if bbox is not None:
            self.obj_bbox[self.nextObjectID] = bbox
        self.nextObjectID += 1

    def deregister(self, object_id):
        del self.obj_centroids[object_id]
        del self.obj_absence_counts[object_id]
        del self.obj_distance_counts[object_id]
        if object_id in self.obj_bbox:
            del self.obj_bbox[object_id]

    def update(self, input_centroids, input_distances, input_bbox=None):
        if len(input_centroids) == 0:
            # if no objects tracked currently
            # inc absence count for previously tracked objects
            for objectID in list(self.obj_absence_counts.keys()):
                self.obj_absence_counts[objectID] += 1

                if self.obj_absence_counts[objectID] > self.max_frame_tracking:
                    self.deregister(objectID)

            return self.obj_centroids

        # if no objects currently tracked take input centroids and register each of them
        if len(self.obj_centroids) == 0:
            if input_bbox is None:
                _ = [
                    self.register(input_centroids[i])
                    for i in range(len(input_centroids))
                ]
            else:
                _ = [
                    self.register(input_centroids[i], input_bbox[i])
                    for i in range(len(input_centroids))
                ]
        # else match the input centroids to existing object centroids
        else:
            objectIDs = list(self.obj_centroids.keys())
            objectCentroids = list(self.obj_centroids.values())

            # compute the distance between each centroids and input centroids pairs
            D = dist.cdist(np.array(objectCentroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()

            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the absence counter
                objectID = objectIDs[row]
                self.obj_centroids[objectID] = input_centroids[col]
                if input_bbox is not None:
                    self.obj_bbox[objectID] = input_bbox[col]
                if input_distances is not None:
                    self.obj_distance_counts[objectID].append(input_distances[col][0])
                self.obj_absence_counts[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # check for disappeared centroids
            # if input centroids number is smaller than existing centroids
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.obj_absence_counts[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.obj_absence_counts[objectID] > self.max_frame_tracking:
                        self.deregister(objectID)

            # else, if input centroids number is greater
            # than existing centroids register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    if input_bbox is None:
                        self.register(input_centroids[col])
                    else:
                        self.register(input_centroids[col], input_bbox[col])

        # return the set of trackable objects
        return self.obj_centroids
