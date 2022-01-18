import json
import os
import sys

import cv2
import numpy as np
import yaml

import cv2utils as cv2utils
import myfunctions as myf
import tracker.deep_sort.nn_matching as nn_matching
import utils as utils
from tracker.application_util import preprocessing
from tracker.application_util.visualization import NoVisualization, Visualization
from tracker.deep_sort.generate_features import create_box_encoder, generate_features
from tracker.deep_sort.tracker import Tracker
from tracker.deep_sort_apgip import create_detections
from yolo_custom import Yolov5


def run(config_data):
    """
    Main function to run video / stream / frames using given (in YAML
    config) algorithms with TensorFlow.
    """

    capture = cv2utils.conditional_capture(config_data)
    model = Yolov5("25.pt")
    model.model_warmup()
    labels = utils.create_coco()

    encoder = create_box_encoder("tracker/deep_sort/mars-small128.pb", batch_size=32)
    metric = nn_matching.NearestNeighborDistanceMetric(
        config_data["metric"], config_data["max_cosine_distance"], None
    )
    tracker = Tracker(
        metric, config_data["camera_params"], config_data["max_iou_distance"]
    )

    seq_info = {
        "sequence_name": "XD",
        "image_size": (1080, 1920),
        "min_frame_idx": 0,
        "max_frame_idx": np.inf,
        "feature_dim": 128,
        "update_ms": 5,
    }

    # visualizer = NoVisualization(seq_info)
    visualizer = Visualization(seq_info, 5)

    def frame_callback(visualizer, frame_number):
        nonlocal labels
        annot_id = len(labels["annotations"]) + 1
        try:
            frame = cv2utils.read_frame(config_data, frame_number, capture)
        except IndexError:
            cv2.destroyAllWindows()
            print("no more frames")
            sys.exit(1)
        if frame is None:
            print("Break due to the video / stream error or end.")
            sys.exit(1)

        print(f"Frame number: {frame_number}")

        objects_in_single_frame = model.inference(frame_number, frame)
        tracing_objects = myf.update_objects_trackers_kalman(frame, tracker.tracks)
        for obj in tracing_objects:
            objects_in_single_frame.append(obj)
        detections = create_detections(objects_in_single_frame, frame_number)
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, config_data["nms_max_overlap"], scores
        )
        detections = [detections[i] for i in indices]
        detections = generate_features(encoder, frame, detections)
        tracker.predict()
        tracker.update(detections, frame.shape, frame_number)
        visualizer.set_image(frame.copy())

        # visualizer.draw_detections(detections)
        visualizer.draw_trackers(tracker.tracks)

        labels = utils.add_to_coco_frames_kalman(
            labels, frame, tracker.tracks, frame_number, config_data, capture, annot_id
        )
        if (
            config_data["type_of_source"] == "frames"
            and frame_number + 1 == len(capture)
        ) or (
            config_data["type_of_source"] == "video"
            and frame_number + 5 == int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ):
            annotations_file_name = (
                f"coco_kalman_{os.path.basename(config_data['path_to_source'])}.json"
            )
            save_path = os.path.join(
                os.path.dirname(config_data["path_to_source"]),
                "new_" + annotations_file_name,
            )
            with open(save_path, "w") as annotation_file:
                json.dump(labels, annotation_file)
            print(f"Saved annotations to: {save_path}")
            sys.exit(1)

    visualizer.run(frame_callback)


if __name__ == "__main__":
    with open("config.yaml") as f:
        f = yaml.load(f, Loader=yaml.FullLoader)
    run(f)
