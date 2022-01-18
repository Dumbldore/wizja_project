from scipy.optimize import curve_fit  # equal to the lookback
from util import func


class Tracker:
    def __init__(self, track_id):
        self.id = track_id
        self.tracks = []
        self.category = -1
        self.poptx = 0
        self.popty = 0
        self.accident = -1

    def average_size(self):
        x = 0
        y = 0
        for i in range(1, 3):
            x += self.tracks[-i].x_size
            y += self.tracks[-i].y_size
        return int(x / 3), int(y / 3)

    def check_category(self):
        categories = [0] * 100
        for track in self.tracks:
            try:
                categories[track.category] += 1
            except:
                pass
        self.category = categories.index(max(categories))

    def fit_line(self):
        x = [track.x for track in self.tracks]
        y = [track.y for track in self.tracks]
        z = [track.frame for track in self.tracks]
        self.poptx, pcovx = curve_fit(func, z, x, maxfev=100000)
        self.popty, pcovy = curve_fit(func, z, y, maxfev=100000)

    def connect_trackers(self, tracker):
        for track in tracker.tracks:
            if track.frame > self.tracks[-1].frame:
                self.tracks.append(track)
        self.fit_line()


class Track:
    def __init__(self, frame, loc_x, loc_y, x_size, y_size, category):
        self.frame = frame
        self.x = loc_x
        self.y = loc_y
        self.y_size = y_size
        self.x_size = x_size
        self.category = category
