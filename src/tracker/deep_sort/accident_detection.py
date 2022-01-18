import math
from cmath import phase, rect
from itertools import combinations
from math import degrees, radians

import numpy as np

# def check_accident(tracks):
#     tracks = [track for track in tracks if track.time_since_update == 0 and len(track.velocity_deque) > 10]
#     overlapped = check_overlaps(tracks)
#     for overlap in overlapped:
#         check_angle(overlap)
#         check_angle(overlap.reverse())
#
#
# def check_angle(overlap):
#     center1 = overlap[0].curr_xywhl[0:2]
#     center2 = overlap[1].curr_xywhl[0:2]
#     dist_x = center2[0] - center1[0]
#     dist_y = center2[1] - center1[1]
#     angle = math.atan2(dist_y, dist_x) / math.pi * 180
#     if angle < 0:
#         angle += 360
#     car_angle = overlap[0].curr_angle
#     if car_angle < 0:
#         car_angle += 360
#     angle_diff = abs(car_angle - angle)
#     if angle_diff > 180:
#         angle_diff = 360 - angle_diff
#     if angle_diff < 90:
#         overlap[0].accident_deque.append((90 - angle_diff) / 90)


class AccidentDetector:
    def __init__(self):
        self.possible_accidents = []

    def run(self, tracks, frame):
        tracks = [track for track in tracks if track.time_since_update == 0]
        overlapped = self.check_overlaps(tracks)
        for overlap in overlapped:
            if all(
                accident.participants != overlap for accident in self.possible_accidents
            ):
                accident = Accident(overlap, frame)
                self.possible_accidents.append(accident)
        for accident in self.possible_accidents:
            accident.update(frame)

    @staticmethod
    def check_overlaps(tracks):
        overlapped_list = []
        for r1, r2 in combinations(tracks, 2):
            r1_bbox = r1.to_tlbr()
            r2_bbox = r2.to_tlbr()
            if r1_bbox[0] > r2_bbox[2] or r1_bbox[2] < r2_bbox[0]:
                continue
            if r1_bbox[3] < r2_bbox[1] or r1_bbox[1] > r2_bbox[3]:
                continue
            if r1.track_id < r2.track_id:
                overlapped_list.append([r1, r2])
            else:
                overlapped_list.append([r2, r1])
        return overlapped_list


class Accident:
    def __init__(self, participants, frame):
        self.participants = participants
        self.start_frame = frame
        self.avg_speed = [None, None]
        self.avg_angle = [None, None]
        self.speed_diff = [None, None]
        self.angle_diff = [None, None]
        self.count_avg_both()

    def average_angle(self, angles):

        return (
            degrees(phase(sum(rect(1, radians(d)) for d in angles) / len(angles))) + 180
        )

    def subtract_angles(self, lhs, rhs):
        diff = lhs - rhs
        if diff > 180:
            diff = 360 - diff
        return diff

    def count_avg_both(
        self,
    ):
        for i, participant in enumerate(self.participants):
            if len(participant.velocity_deque) < 10:
                continue
            self.avg_speed[i] = np.mean(participant.velocity_deque)
            self.avg_angle[i] = self.average_angle(participant.fixed_angle_deque360)

    def count_avg(self, i):
        if len(self.participants[i].velocity_deque) < 10:
            self.avg_speed[i] = None
            return
        self.avg_speed[i] = np.mean(self.participants[i].velocity_deque)
        if self.avg_speed[i] < 9:
            self.avg_angle[i] = 0
        else:
            self.avg_angle[i] = self.average_angle(
                self.participants[i].fixed_angle_deque360
            )

    def compare_avg(self, frames):
        for i, participant in enumerate(self.participants):
            if self.avg_speed[i] != None and self.speed_diff[i] != None:
                new_avg_speed = np.mean(list(participant.velocity_deque)[-frames:])
                if self.avg_angle[i] == 0:
                    new_avg_angle = 0
                else:
                    new_avg_angle = self.average_angle(
                        list(participant.fixed_angle_deque360)[-frames:]
                    )
                self.speed_diff[i] = abs(self.avg_speed[i] - new_avg_speed)
                self.angle_diff[i] = abs(
                    self.subtract_angles(self.avg_angle[i], new_avg_angle)
                )
                if (
                    self.speed_diff[i] > 12.5
                    and self.angle_diff[i] > 45
                    or self.speed_diff[i] > 20
                ):
                    print(
                        f"Track {self.participants[0].track_id} and {self.participants[1].track_id}"
                    )
                    print(f"Angle sum {self.angle_diff[i]}")
                    print(f"Speed sum {self.speed_diff[i]}")
                    self.participants[0].accident = 1
                    self.participants[1].accident = 1

            else:
                self.speed_diff[i] = 0
                self.angle_diff[i] = 0
                self.count_avg(i)
        # if sum(self.speed_diff) > 12.5 and sum(self.angle_diff) > 45 or sum(self.speed_diff) > 20:
        #     print(f"Track {self.participants[0].track_id} and {self.participants[1].track_id}")
        #     print(f"Angle sum {sum(self.angle_diff)}")
        #     print(f"Speed sum {sum(self.speed_diff)}")
        #     self.participants[0].accident = 1
        #     self.participants[1].accident = 1

    def update(self, frame):
        if 50 > frame - self.start_frame > 10:
            self.compare_avg(frame - self.start_frame)
