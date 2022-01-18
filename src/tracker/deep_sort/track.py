# vim: expandtab:ts=4:sw=4
import collections
import math

import numpy as np

from .default_sizes import (
    default_objects_sizes,
    max_objects_sizes,
    min_objects_sizes,
    vehicles,
    vertical_objects,
)


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(
        self,
        mean,
        covariance,
        track_id,
        n_init,
        max_age,
        max_cv_tracker_age,
        feature=None,
        vel_deque_max_len=15,
        angle_deque_max_len=15,
        xyz_positions_deque_max_len=30,
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.accident = 0
        self.age = 1
        self.time_since_update = 0
        self.cv_tracker = None
        self.cv_track_age = 1
        self.label = None
        self.state = TrackState.Tentative
        self.last_bbox = None
        self.last_frame = 0
        self.curr_velocity = None
        self.curr_angle = None
        self.velocity_deque = collections.deque(maxlen=vel_deque_max_len)
        self.raw_angle_deque = collections.deque(maxlen=angle_deque_max_len)
        self.fixed_angle_deque = collections.deque(maxlen=xyz_positions_deque_max_len)
        self.fixed_angle_deque360 = collections.deque(
            maxlen=xyz_positions_deque_max_len
        )
        self.timestamps_deque = collections.deque(maxlen=xyz_positions_deque_max_len)
        self.x_position_deque = collections.deque(maxlen=xyz_positions_deque_max_len)
        self.y_position_deque = collections.deque(maxlen=xyz_positions_deque_max_len)
        self.width_deque = collections.deque(maxlen=xyz_positions_deque_max_len)
        self.height_deque = collections.deque(maxlen=xyz_positions_deque_max_len)
        self.length_deque = collections.deque(maxlen=xyz_positions_deque_max_len)
        self.curr_xywhl = None
        self.last_xywhl = None
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        self._max_age = max_age
        self._max_cv_tracker_age = max_cv_tracker_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection, camera_params, img_shape, frame_number):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        if detection.confidence == 0.4:
            self.cv_track_age += 1
        else:
            self.cv_track_age = 1
        if self.cv_track_age >= self._max_cv_tracker_age:
            self.mark_missed()
        else:
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, detection.to_xyah()
            )
            self.features.append(detection.feature)
            self.label = detection.label
            self.hits += 1
            self.time_since_update = 0
            self.curr_xywhl = self._calc_position_and_size(camera_params, img_shape)
            (
                self.curr_velocity,
                self.curr_angle,
            ) = self._calc_vel_and_angle_from_position(frame_number)
            self._correct_xywhl(frame_number, camera_params["camera_height"])
            self.last_frame = frame_number
            self.last_xywhl = self.curr_xywhl
            if self.state == TrackState.Tentative and self.hits >= self._n_init:
                self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def _calc_position_and_size(self, cam_params, img_shape):
        """
        Estimating object's X and Y coordinates relative to the camera and real object's height and width based on
        bounding box. Distance to camera is calculated from the middle of bottom edge of bounding box.
        :param cam_params: Camera parameters specified in config.yaml
        :param img_shape: Image resolution (image height x image width)
        :return: Object's X and Y distance to camera and object's real height and width in meters
        """

        # Getting coordinates of left top corner of bounding box
        left, top, width, height = list(map(int, self.to_tlwh().tolist()))
        # Getting image width and height
        image_height, image_width, _ = img_shape

        # Estimating x,y and distance to object in straight line

        # Calculating object's position in Y axis in pixels
        j_bot = top + height
        # Calculating psi angle for object's position in Y axis
        psi_bot = cam_params["pitch"] + (image_height / 2 - j_bot) * (
            cam_params["fov_vertical"] / image_height
        )
        # Calculating distance im meters for object's position in Y axis
        y_bot = cam_params["camera_height"] * math.tan(math.radians(psi_bot))
        # Calculating object's position in X axis in pixels
        i_mid = left + width / 2
        # Calculating fi angle for object's position in X axis
        fi_mid = (i_mid - image_width / 2) * (
            cam_params["fov_horizontal"] / image_width
        )
        # Calculating distance im meters for object's position in X axis
        x_bot = y_bot * math.tan(math.radians(fi_mid))

        # Estimating object's height

        # Calculating object's top edge position in Y axis in pixels
        j_top = top
        # Calculating psi angle for object's top edge position in Y axis
        psi_top = cam_params["pitch"] + (image_height / 2 - j_top) * (
            cam_params["fov_vertical"] / image_height
        )
        # Calculating distance im meters for shadow of object's top edge in Y axis
        y_top = cam_params["camera_height"] * math.tan(math.radians(psi_top))
        # Calculating object's height in meters
        object_height = cam_params["camera_height"] * ((y_top - y_bot) / y_top)

        # Estimating object's width

        # Calculating positions in meters in Y axis of object's right and left edge
        y_right = y_left = y_bot
        # Calculating positions in pixels in X axis of object's right and left edge
        i_left = left
        i_right = left + width
        # Calculating fi angles for positions of object's right and left edge in X axis
        fi_left = (i_left - image_width / 2) * (
            cam_params["fov_horizontal"] / image_width
        )
        fi_right = (i_right - image_width / 2) * (
            cam_params["fov_horizontal"] / image_width
        )
        # Calculating distances im meters for object's right and left edge in X axis
        x_left = y_left * math.tan(math.radians(fi_left))
        x_right = y_right * math.tan(math.radians(fi_right))
        # Calculating object's width
        object_width = math.sqrt((x_right - x_left) ** 2 + (y_right - y_left) ** 2)

        return [x_bot, y_bot, object_width, object_height, -1]

    def _calc_vel_and_angle_from_position(self, frame_number):
        """
        Calculate object's velocity and the angle of his trajectory. Velocity and angle are calculated based on last
        'use_last_n' measurements. Firstly, outliers from these measurements are removed, then the final values are the average
        from the rest values.
        Angle represents the counterclockwise angle in degrees between the positive X-axis and the direction in which
        object is heading. Range from -180 to 180 where 0=East, -90=North, 90=South and -180/180 = West
        :param frame_number: Frame id
        :return: Velocity in km/h and angle with degrees.
        """
        if self.last_xywhl is None:
            self.fixed_angle_deque.append(0)
            self.fixed_angle_deque360.append(0)
            return 0, 0

        # Getting object's position in XY axis and its size in last frame
        center1 = self.last_xywhl[0:2]
        # Getting object's position in XY axis and its size in current frame
        center2 = self.curr_xywhl[0:2]
        # Calculating partial distances between object's positions in X and Y axis
        dist_x = center2[0] - center1[0]
        dist_y = center2[1] - center1[1]
        # Calculating total distance between object's positions
        distance = math.sqrt(math.pow(dist_x, 2) + math.pow(dist_y, 2))
        # Calculating number of frames between current and last position
        life_time = frame_number - self.last_frame
        # Calculating object's velocity in km/h
        velocity_km_h = (
            distance / (life_time / 30) * 3.6
        )  # Divided by 30 because video is in 30 FPS

        # The number returned represents the counterclockwise angle in degrees between the positive X-axis and the
        # point (x, y).
        angle = math.atan2(dist_y, dist_x) / math.pi * 180

        fixed_angle = self._rm_outliers_and_calc_angle(angle, use_last_n=5)
        self.fixed_angle_deque.append(fixed_angle)
        self.fixed_angle_deque360.append(fixed_angle + 180)
        return (
            self._rm_outliers_and_calc_velocity(velocity_km_h, use_last_n=5),
            fixed_angle,
        )

    def _rm_outliers_and_calc_velocity(self, curr_velocity_km_h, use_last_n=10):
        """
        Removes outliers from the last values of velocity and then calculates new value of velocity
        by calculating mean from 'use_last_n' left velocities.
        :return: Velocity in km/h
        """

        # If velocity has increased by 50% in comparison with the mean value from the previous values,
        # then it is probably an outlier, and it shouldn't be counted as a valid measurement.
        # Additional conditions: deque is full, mean velocity is higher than 25 km/h
        if not (
            len(self.velocity_deque) == self.velocity_deque.maxlen
            and np.mean(self.velocity_deque) > 25
            and curr_velocity_km_h > 1.5 * np.mean(self.velocity_deque)
        ):
            # Adding current velocity to list of valid past velocities
            self.velocity_deque.append(curr_velocity_km_h)

        # Value of maximum deviation is set to 1.5
        max_deviation = 1.5
        # Converting list of valid past velocities to numpy array format
        velocity_data = np.array(self.velocity_deque)
        # Calculating deviation for each past velocity
        deviation = np.abs(velocity_data - np.median(velocity_data))
        # Calculating median from deviations
        median_deviation = np.median(deviation)
        # Calculating filtering factor and filtering numpy array with past velocities
        s = deviation / median_deviation if median_deviation else 1.0
        filtered_velocity_data = velocity_data[s < max_deviation]
        # Returning mean from filtered velocities or mean from not-filtered velocities if list of filtered velocites is empty
        return (
            np.mean(filtered_velocity_data[-use_last_n:])
            if filtered_velocity_data != []
            else np.mean(velocity_data[-use_last_n:])
        )

    def _rm_outliers_and_calc_angle(self, curr_angle, use_last_n=5):
        """
        Removes outliers from the last values of angle and then calculates new value of angle
        by calculating mean from 'use_last_n' left angles.
        :return: Angle in degrees
        """
        self.raw_angle_deque.append(curr_angle)
        max_deviation = 3
        angle_data = np.array(self.raw_angle_deque)
        filtered_angle_data = angle_data[
            abs(angle_data - np.mean(angle_data)) <= max_deviation * np.std(angle_data)
        ]
        return (
            np.mean(filtered_angle_data[-use_last_n:])
            if filtered_angle_data != []
            else np.mean(angle_data[-use_last_n:])
        )

    def _correct_xywhl(self, frame_number, camera_height):
        """
        Improves values of object's width, height and lenght using its precise orientation in X,Y,Z axis.
        :param frame_number: Frame id
        :param camera_height: Camera height in meters
        :return:
        """
        rotate_angle = abs(self.curr_angle)
        if rotate_angle > 90:
            rotate_angle = 180 - rotate_angle
        rotate_angle_radians = math.radians(rotate_angle)
        # Pitch: Range from -180 to +180 where 0 = horizon, +180 = straight up and –180 = straight down (relative to
        # camera)

        dist_y_from_object_to_camera = self.curr_xywhl[1]
        pitch = math.degrees(math.atan2(camera_height, dist_y_from_object_to_camera))

        real_width = None
        real_length = None
        real_height = None

        old_width = self.curr_xywhl[2]
        old_height = self.curr_xywhl[3]

        known_objects = default_objects_sizes.keys()
        if self.label in known_objects:
            # Calculations for width and length
            if self.label in vehicles:
                if 0 <= rotate_angle <= 20:
                    real_length = math.cos(rotate_angle_radians) * old_width
                    default_sizes = default_objects_sizes[self.label]
                    real_width = real_length * default_sizes[1] / default_sizes[0]
                    obj_size_in_y_axis = real_width
                elif 20 < rotate_angle <= 30:
                    real_length = math.cos(rotate_angle_radians) * old_width
                    real_width = math.sin(rotate_angle_radians) * old_width
                    obj_size_in_y_axis = math.sin(rotate_angle_radians) * real_length
                elif 30 < rotate_angle <= 55:
                    bbox_left, bbox_top, bbox_width, bbox_height = list(
                        map(int, self.to_tlwh().tolist())
                    )

                    beta_radians = math.radians(90 - rotate_angle)
                    pixel_segment_len = math.tan(beta_radians) * bbox_height / 2
                    meters_segment_len = old_width * pixel_segment_len / bbox_width
                    real_length = 2 * (meters_segment_len / math.sin(beta_radians))

                    real_width = 2 * (math.cos(beta_radians) * meters_segment_len)
                    obj_size_in_y_axis = 2 * meters_segment_len / math.tan(beta_radians)
                elif 55 <= rotate_angle <= 90:
                    real_width = math.sin(rotate_angle_radians) * old_width
                    default_sizes = default_objects_sizes[self.label]
                    real_length = real_width * default_sizes[0] / default_sizes[1]
                    obj_size_in_y_axis = real_length
                else:
                    real_width = old_width
                    real_length = old_width
                    obj_size_in_y_axis = math.sqrt(2) / 2 * real_length

                # Calculations for height
                if 0 <= abs(pitch) <= 20:
                    real_height = old_height
                else:
                    bbox_shadow_len = (old_height * self.curr_xywhl[1]) / (
                        camera_height - old_height
                    )
                    objects_shadow_len = bbox_shadow_len - obj_size_in_y_axis
                    real_height = old_height * objects_shadow_len / bbox_shadow_len

            elif self.label in vertical_objects:
                real_height = old_height
                real_width = old_width
                real_length = old_width
            else:
                real_height = old_height
                real_width = old_width
                real_length = old_width

            # Validate results
            if (
                real_length > max_objects_sizes[self.label][0]
                or real_length < min_objects_sizes[self.label][0]
                or real_width > max_objects_sizes[self.label][1]
                or real_width < min_objects_sizes[self.label][1]
            ):
                real_length, real_width = default_objects_sizes[self.label][0:2]

            if (
                real_height > max_objects_sizes[self.label][2]
                or real_height < min_objects_sizes[self.label][2]
            ):
                real_height = default_objects_sizes[self.label][2]

        else:
            real_height = old_height
            real_width = old_width
            real_length = old_width

        self.curr_xywhl[2] = real_width
        self.curr_xywhl[3] = real_height
        self.curr_xywhl[4] = real_length

        self.timestamps_deque.append(
            frame_number
        )  # TODO: Replace frame number with timestamp from Deepstream
        self.x_position_deque.append(self.curr_xywhl[0])
        self.y_position_deque.append(self.curr_xywhl[1])
        self.width_deque.append(self.curr_xywhl[2])
        self.height_deque.append(self.curr_xywhl[3])
        self.length_deque.append(self.curr_xywhl[4])

    def get_predicted_corridor(self, how_many_future_frames=60):
        """
        Calculating 3D corridor representing probable future trajectory of track.
        :param how_many_future_frames: How many positions in the future should be calculated
        :return: Dict containing 3D edges of predicted corridor
        """
        if len(self.x_position_deque) != self.x_position_deque.maxlen:
            return None

        predicted_corridor = {}

        time_train = np.array(self.timestamps_deque)
        time_predicted = np.array(
            range(time_train[-1] + 1, time_train[-1] + how_many_future_frames, 1)
        )  # TODO: Replace frame number with timestamp from Deepstream
        """
        Second approach: Calculate all points locations based on mid bottom point and all past widths, heights, lengths and 
        orientations
        """
        x_position_deque_np = np.array(self.x_position_deque)
        y_position_deque_np = np.array(self.y_position_deque)
        length_deque_np = np.array(self.length_deque)
        width_deque_np = np.array(self.width_deque)

        left_front_points, right_front_points = [
            x_position_deque_np + length_deque_np / 2,
            y_position_deque_np,
        ], [
            x_position_deque_np + length_deque_np / 2,
            y_position_deque_np + width_deque_np,
        ]
        object_centroids = [
            x_position_deque_np,
            y_position_deque_np + width_deque_np / 2,
        ]

        fixed_angle_radians = np.radians(self.fixed_angle_deque)
        sinus_array = np.sin(fixed_angle_radians)
        cosinus_array = np.cos(fixed_angle_radians)

        x_left_front_points = left_front_points[0]  # Copy
        y_left_front_points = left_front_points[1]  # Copy
        left_front_points[0] = (
            object_centroids[0]
            + cosinus_array * (x_left_front_points - object_centroids[0])
            - sinus_array * (y_left_front_points - object_centroids[1])
        )
        left_front_points[1] = (
            object_centroids[1]
            + sinus_array * (x_left_front_points - object_centroids[0])
            + cosinus_array * (y_left_front_points - object_centroids[1])
        )

        x_right_front_points = right_front_points[0]  # Copy
        y_right_front_points = right_front_points[1]  # Copy
        right_front_points[0] = (
            object_centroids[0]
            + cosinus_array * (x_right_front_points - object_centroids[0])
            - sinus_array * (y_right_front_points - object_centroids[1])
        )
        right_front_points[1] = (
            object_centroids[1]
            + sinus_array * (x_right_front_points - object_centroids[0])
            + cosinus_array * (y_right_front_points - object_centroids[1])
        )

        x_train_left = left_front_points[0]
        x_train_right = right_front_points[0]
        y_train_left = left_front_points[1]
        y_train_right = right_front_points[1]
        x_train_mid = (x_train_left + x_train_right) / 2

        x_predicted_left = self._extrapolate_trajectory(
            time_train, time_predicted, x_train_left
        )
        x_predicted_right = self._extrapolate_trajectory(
            time_train, time_predicted, x_train_right
        )
        x_predicted_mid = self._extrapolate_trajectory(
            time_train, time_predicted, x_train_mid
        )

        y_predicted_left = self._extrapolate_trajectory(
            time_train, time_predicted, y_train_left
        )
        y_predicted_right = self._extrapolate_trajectory(
            time_train, time_predicted, y_train_right
        )

        z_train_down = np.zeros(x_train_left.shape)
        z_train_up = np.array(self.height_deque)
        z_predicted_down = self._extrapolate_trajectory(
            time_train, time_predicted, z_train_down
        )
        z_predicted_up = self._extrapolate_trajectory(
            time_train, time_predicted, z_train_up
        )

        # Object's coordinates with assumption that object is pointing straight to the right
        (
            (x_left_back, y_left_back),
            (x_left_front, y_left_front),
            (x_right_front, y_right_front),
            (x_right_back, y_right_back),
        ) = self.get_track_vertices()

        x_left_back, y_left_back = np.array([x_left_back]), np.array([y_left_back])
        x_left_front, y_left_front = np.array([x_left_front]), np.array([y_left_front])
        x_right_back, y_right_back = np.array([x_right_back]), np.array([y_right_back])
        x_right_front, y_right_front = np.array([x_right_front]), np.array(
            [y_right_front]
        )
        curr_height = np.array([self.curr_xywhl[3]])

        # Extend corridor by the object cubic
        x_predicted_left = np.concatenate([x_left_back, x_left_front, x_predicted_left])
        x_predicted_right = np.concatenate(
            [x_right_back, x_right_front, x_predicted_right]
        )
        x_predicted_mid = np.concatenate(
            [
                (x_left_back + x_right_back) / 2,
                (x_left_front + x_right_front) / 2,
                x_predicted_mid,
            ]
        )
        y_predicted_left = np.concatenate([y_left_back, y_left_front, y_predicted_left])
        y_predicted_right = np.concatenate(
            [y_right_back, y_right_front, y_predicted_right]
        )
        z_predicted_down = np.concatenate(
            [np.array([0]), np.array([0]), z_predicted_down]
        )
        z_predicted_up = np.concatenate([curr_height, curr_height, z_predicted_up])

        predicted_corridor["bot_left"] = {
            "x": x_predicted_left,
            "y": y_predicted_left,
            "z": z_predicted_down,
        }

        predicted_corridor["bot_right"] = {
            "x": x_predicted_right,
            "y": y_predicted_right,
            "z": z_predicted_down,
        }

        predicted_corridor["top_left"] = {
            "x": x_predicted_left,
            "y": y_predicted_left,
            "z": z_predicted_up,
        }

        predicted_corridor["top_right"] = {
            "x": x_predicted_right,
            "y": y_predicted_right,
            "z": z_predicted_up,
        }

        predicted_corridor["top_mid"] = {
            "x": x_predicted_mid,
            "y": (y_predicted_left + y_predicted_right) / 2,
            "z": z_predicted_up,
        }

        return predicted_corridor

    def _extrapolate_trajectory_3D(
        self, time_train, time_predicted, x_train, y_train, z_train
    ):
        """
        Extrapolated trajectory for given points in 3D
        :param time_train: Time values used to train model
        :param time_predicted: Time values for which 3D trajectory will be predicted
        :param x_train: Values on X axis used to train model
        :param y_train: Values on Y axis used to train model
        :param z_train: Values on Z axis used to train model
        :return: X, Y and Z values for 'time_predicted'
        """
        f_poly1d_x = np.poly1d(np.polyfit(time_train, x_train, 1))
        f_poly1d_y = np.poly1d(np.polyfit(time_train, y_train, 1))
        f_poly1d_z = np.poly1d(np.polyfit(time_train, z_train, 1))

        x_predicted = f_poly1d_x(time_predicted)
        y_predicted = f_poly1d_y(time_predicted)
        z_predicted = f_poly1d_z(time_predicted)

        return x_predicted, y_predicted, z_predicted

    def _extrapolate_trajectory(self, time_train, time_predicted, values):
        """
        Extrapolated future trajectory based on past values
        :param time_train: Time values used to train model
        :param time_predicted: Time values for which trajectory will be predicted
        :param values: Y values used to train model
        :return: Values predicted for 'time_predicted'
        """
        f_poly1d_x = np.poly1d(np.polyfit(time_train, values, 1))
        values_predicted = f_poly1d_x(time_predicted)
        return values_predicted

    def get_track_vertices(self):
        track_x, track_y, track_width, track_height, track_length = self.curr_xywhl
        # Object's coordinates with assumption that object is pointing straight to the right
        pt_left_back, pt_left_front, pt_right_front, pt_right_back = (
            (track_x - track_length / 2, track_y),
            (track_x + track_length / 2, track_y),
            (track_x + track_length / 2, track_y + track_width),
            (track_x - track_length / 2, track_y + track_width),
        )

        # self.curr_angle: Range from -180 to 180 where 0=East, -90=South, 90=North and -180/180 = West
        # Yaw: Range from -180 to 180 where 0=East, 90=South, -90=North and -180/180 = West
        yaw = -self.curr_angle
        pt_left_back, pt_left_front, pt_right_front, pt_right_back = self._rotate_track(
            pt_left_back, pt_left_front, pt_right_front, pt_right_back, yaw
        )
        return pt_left_back, pt_left_front, pt_right_front, pt_right_back

    def _rotate_track(
        self, pt_left_back, pt_left_front, pt_right_front, pt_right_back, yaw
    ):
        """
        Rotate 4 vertices counterclockwise relative to the center with chosen angle
        :param yaw: Rotation angle in degrees
        :return: 4 rotated vertices
        """
        rotate_angle = -yaw
        centroid = (
            (pt_left_back[0] + pt_left_front[0]) / 2,
            (pt_left_back[1] + pt_right_back[1]) / 2,
        )

        rotate_angle_radians = math.radians(rotate_angle)
        pt_left_back = self._rotate_point(centroid, pt_left_back, rotate_angle_radians)
        pt_left_front = self._rotate_point(
            centroid, pt_left_front, rotate_angle_radians
        )
        pt_right_front = self._rotate_point(
            centroid, pt_right_front, rotate_angle_radians
        )
        pt_right_back = self._rotate_point(
            centroid, pt_right_back, rotate_angle_radians
        )

        return pt_left_back, pt_left_front, pt_right_front, pt_right_back

    @staticmethod
    def _rotate_point(anchor, rotated_point, rotate_angle_radians):
        """
        Rotate a point counterclockwise by a given angle around a given anchor.
        The angle should be given in radians.
        """
        ox, oy = anchor
        px, py = rotated_point
        rotated_x = (
            ox
            + math.cos(rotate_angle_radians) * (px - ox)
            - math.sin(rotate_angle_radians) * (py - oy)
        )
        rotated_y = (
            oy
            + math.sin(rotate_angle_radians) * (px - ox)
            + math.cos(rotate_angle_radians) * (py - oy)
        )
        return rotated_x, rotated_y
