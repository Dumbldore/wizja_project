import json

from tracker import Track, Tracker


def load_trackers(path):
    with open(path) as f:
        coco = json.loads(f.read())

    trackers = {}
    for annotation in coco["annotations"]:
        if not any(
            trackers[tracker].id == annotation["track_id"] for tracker in trackers
        ):
            trackers[annotation["track_id"]] = Tracker(annotation["track_id"])

    def bbox_center(bbox):
        return [int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)]

    for annotation in coco["annotations"]:
        if annotation["track_id"] in trackers:
            trackers[annotation["track_id"]].tracks.append(
                Track(
                    annotation["image_id"],
                    bbox_center(annotation["bbox"])[0],
                    bbox_center(annotation["bbox"])[1],
                    int(annotation["bbox"][2]),
                    int(annotation["bbox"][3]),
                    annotation["category_id"],
                )
            )

    for key, tracker in list(trackers.items()):
        if len(tracker.tracks) < 10:
            del trackers[key]
            continue
        tracker.check_category()
        tracker.fit_line()
    trackers = list(trackers.values())
    return trackers
