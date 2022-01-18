import os
import re
import sys

import cv2


def conditional_capture(config_data):
    """
    Based on YAML config cv2.VideoCapture is created to obtain frames
    from it. If source type is 'frames' rather than 'video' capture is
    basically None.
    NOTE: Not sure what about stream yet.
    """
    if config_data["type_of_source"] == "frames":
        files = [
            f
            for f in os.listdir(config_data["path_to_source"])
            if os.path.isfile(os.path.join(config_data["path_to_source"], f))
        ]
        r = re.compile("\d+")
        files.sort(key=lambda x: int(r.search(x).group()))
        return files
    elif config_data["type_of_source"] == "video":
        return cv2.VideoCapture(config_data["path_to_source"])
    else:
        exit(1)


def read_frame(config_data, current_frame_number, capture):
    """
    Read next frame from dir if source is 'frames' or from video if
    source is 'video'.
    """

    if config_data["type_of_source"] == "frames":
        """
        NOTE: image_name convention is to be implemented!
        """
        frame = cv2.imread(
            os.path.join(config_data["path_to_source"], capture[current_frame_number]),
            cv2.IMREAD_COLOR,
        )
        return frame

    elif config_data["type_of_source"] == "video" and capture is not None:
        _, frame = capture.read()
        return frame

    else:
        sys.exit(1)
