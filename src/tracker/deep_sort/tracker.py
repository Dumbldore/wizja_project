# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

import numpy as np
from shapely.geometry import Polygon

from . import iou_matching, kalman_filter, linear_assignment
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(
        self,
        metric,
        camera_params,
        max_iou_distance=20,
        max_age=20,
        n_init=2,
        max_cv_tracker_age=15,
    ):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = kalman_filter.KalmanFilter()
        self.cam_params = camera_params
        self.tracks = []
        self._next_id = 1
        self.max_cv_tracker_age = max_cv_tracker_age

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, img_shape, frame_number):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf,
                detections[detection_idx],
                self.cam_params,
                img_shape,
                frame_number,
            )
        for track_idx in unmatched_tracks:

            # Count IoU between chosen track and corridor of other chosen track:
            # for matched_track_idx, matched_detection_idx in matches:
            #
            #     #TODO: Replace target track with real missed tracks
            #     IoU, G_IoU = self._calculate_IoU(known_track=self.tracks[matched_track_idx],
            #                                      unrecognized_track=self.tracks[matched_track_idx])

            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features.
        (
            matches_a,
            unmatched_tracks_a,
            unmatched_detections,
        ) = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        (
            matches_b,
            unmatched_tracks_b,
            unmatched_detections,
        ) = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self._next_id,
                self.n_init,
                self.max_age,
                self.max_cv_tracker_age,
                detection.feature,
            )
        )
        self._next_id += 1

    def _calculate_IoU(self, known_track, unrecognized_track):
        """
        Calculate IoU from cuboid created from chosen track and 3D predicted corridor created based on past trajectory
        of another track.
        :param known_track: Track based on which predicted corridor will be calculated
        :param unrecognized_track: Track matched into corridor by calculating its cuboid and IoU with corridor
        :return: IoU in 3D, Generalized IoU in 3D
        """

        pred_corridor = known_track.get_predicted_corridor(how_many_future_frames=60)
        if not pred_corridor:
            return 0, -1

        (
            track_x,
            track_y,
            track_width,
            track_height,
            track_length,
        ) = unrecognized_track.curr_xywhl

        # Object's coordinates with assumption that object is pointing straight to the right
        (
            pt_left_back,
            pt_left_front,
            pt_right_front,
            pt_right_back,
        ) = unrecognized_track.get_track_vertices()

        track_polygon_2d = Polygon(
            [pt_left_back, pt_left_front, pt_right_front, pt_right_back]
        )

        object_last_pos_polygon_2d = Polygon(
            [
                (pred_corridor["bot_left"]["x"][0], pred_corridor["bot_left"]["y"][0]),
                (pred_corridor["bot_left"]["x"][1], pred_corridor["bot_left"]["y"][1]),
                (
                    pred_corridor["bot_right"]["x"][1],
                    pred_corridor["bot_right"]["y"][1],
                ),
                (
                    pred_corridor["bot_right"]["x"][0],
                    pred_corridor["bot_right"]["y"][0],
                ),
            ]
        )

        corridor_polygon_2d_predicted = Polygon(
            [
                (pred_corridor["bot_left"]["x"][2], pred_corridor["bot_left"]["y"][2]),
                (
                    pred_corridor["bot_left"]["x"][-1],
                    pred_corridor["bot_left"]["y"][-1],
                ),
                (
                    pred_corridor["bot_right"]["x"][-1],
                    pred_corridor["bot_right"]["y"][-1],
                ),
                (
                    pred_corridor["bot_right"]["x"][2],
                    pred_corridor["bot_right"]["y"][2],
                ),
            ]
        )

        volume_track = track_width * track_length * track_height
        volume_last_object_pos = (
            object_last_pos_polygon_2d.area * pred_corridor["top_mid"]["z"][0]
        )
        volume_corridor_pred = corridor_polygon_2d_predicted.area * np.mean(
            pred_corridor["top_mid"]["z"]
        )

        # Calculating for object current position:
        try:
            intersection_last_obj_pos_polygon = track_polygon_2d.intersection(
                object_last_pos_polygon_2d
            )
        except Exception as e:
            print(f"Error while calculating intersection - {e}")
            intersection_last_obj_pos_polygon = None

        if intersection_last_obj_pos_polygon and intersection_last_obj_pos_polygon.area:
            union_last_obj_pos = track_polygon_2d.union(object_last_pos_polygon_2d)
            Intersection2D_last_obj_pos = intersection_last_obj_pos_polygon.area
            Intersection_height_last_obj_pos = min(
                track_height, pred_corridor["top_mid"]["z"][0]
            )
            Intersection_3D_last_obj_pos = (
                Intersection2D_last_obj_pos * Intersection_height_last_obj_pos
            )
        else:
            Intersection_3D_last_obj_pos = 0
            union_last_obj_pos = object_last_pos_polygon_2d

        # Calculating for predicted:
        try:
            intersection_polygon_pred = track_polygon_2d.intersection(
                corridor_polygon_2d_predicted
            )
        except Exception as e:
            intersection_polygon_pred = None

        if intersection_polygon_pred and intersection_polygon_pred.area:

            union_polygon_pred = track_polygon_2d.union(corridor_polygon_2d_predicted)
            Intersection2D_pred = intersection_polygon_pred.area

            (
                min_intersect_x,
                min_intersect_y,
                max_intersect_x,
                max_intersect_y,
            ) = intersection_polygon_pred.bounds

            # Calculating more precise average height of intersected area
            corridor_higher_than_cuboid = []
            corridor_lower_than_cuboid = []
            for cnt in range(2, len(pred_corridor["top_mid"]["z"]), 1):
                if (
                    min_intersect_x
                    <= pred_corridor["top_mid"]["x"][cnt]
                    <= max_intersect_x
                    and min_intersect_y
                    <= pred_corridor["top_mid"]["y"][cnt]
                    <= max_intersect_y
                ):
                    if pred_corridor["top_mid"]["z"][cnt] < track_height:
                        corridor_lower_than_cuboid.append(
                            pred_corridor["top_mid"]["z"][cnt]
                        )
                    else:
                        corridor_higher_than_cuboid.append(track_height)

            total_intersect_len = len(corridor_higher_than_cuboid) + len(
                corridor_lower_than_cuboid
            )
            if not corridor_lower_than_cuboid:
                Intersection_height_pred = track_height
            elif not corridor_higher_than_cuboid:
                Intersection_height_pred = np.mean(corridor_lower_than_cuboid)
            else:
                Intersection_height_pred = (
                    np.mean(corridor_higher_than_cuboid)
                    * len(corridor_higher_than_cuboid)
                    / total_intersect_len
                    + np.mean(corridor_lower_than_cuboid)
                    * len(corridor_lower_than_cuboid)
                    / total_intersect_len
                )

            Intersection_3D_pred = Intersection2D_pred * Intersection_height_pred

        else:
            Intersection_3D_pred = 0
            union_polygon_pred = corridor_polygon_2d_predicted

        Intersection_3D_total = Intersection_3D_last_obj_pos + Intersection_3D_pred
        Union_3D_total = (
            volume_track
            + volume_last_object_pos
            + volume_corridor_pred
            - Intersection_3D_total
        )
        IoU_3D = Intersection_3D_total / Union_3D_total

        union_total = union_last_obj_pos.union(union_polygon_pred)
        (
            x_min_enclosing_pred,
            y_min_enclosing_pred,
            x_max_enclosing_pred,
            y_max_enclosing_pred,
        ) = union_total.bounds
        z_min_enclosing_pred = np.min(
            (
                np.min(
                    (pred_corridor["bot_left"]["z"], pred_corridor["bot_right"]["z"])
                ),
                track_height,
                0,
            )
        )
        z_max_enclosing_pred = np.max(
            (
                np.max(
                    (pred_corridor["top_left"]["z"], pred_corridor["top_right"]["z"])
                ),
                track_height,
                0,
            )
        )

        volume_enclosing_box_pred = (
            (x_max_enclosing_pred - x_min_enclosing_pred)
            * (y_max_enclosing_pred - y_min_enclosing_pred)
            * (z_max_enclosing_pred - z_min_enclosing_pred)
        )

        # Range for G_IoU is <-1, 1>
        Generalized_IoU_3D = (
            IoU_3D
            - (volume_enclosing_box_pred - Union_3D_total) / volume_enclosing_box_pred
        )

        return IoU_3D, Generalized_IoU_3D
