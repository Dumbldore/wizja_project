import os

output_labels = {
    "person": ["pedestrian", "person_sitting", "person"],
    "people": ["pedestrians"],
    "wheelchair": ["wheelchairuser"],
    "bicycle": ["cyclist", "biker", "tricyclist"],
    "car": ["van", "1", "2", "3", "7"],
    "motorcycle": ["motorbike", "mopedrider", "motorcyclist"],
    "aeroplane": [],
    "bus": ["8", "9"],
    "train": [],
    "tram": [],
    "truck": ["4", "5", "6", "big-truck"],
    "boat": [],
    "traffic light": [],
    "fire hydrant": [],
    "stop sign": [],
    "traffic sign": [],
    "parking meter": [],
    "bench": [],
    "bird": [],
    "cat": [],
    "dog": [],
    "horse": [],
    "sheep": [],
    "cow": [],
    "police": [],
    "non_motorized_vehicle": ["10"],
}


def create_coco():
    labels = {
        "images": [],
        "categories": generate_categories(output_labels),
        "annotations": [],
    }
    return labels


def generate_categories(expected_labels):
    categories = []
    for i, label in enumerate(expected_labels):
        curr_cat = {"id": i + 1, "name": label, "supercategory": ""}
        categories.append(curr_cat)
    return categories


def add_to_coco_frames_kalman(
    labels, frame, tracks, img_counter, config_data, capture, annot_id
):
    new_image = {
        "id": img_counter,
        "dataset_id": None,
        "path": os.path.join(config_data["path_to_source"], str(img_counter)),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "file_name": str(img_counter),
    }
    labels["images"].append(new_image)
    for track in list(
        filter(lambda x: x.state == 2 and x.time_since_update == 0, tracks)
    ):
        new_detection = {"id": annot_id}
        new_detection["track_id"] = track.track_id
        new_detection["image_id"] = img_counter
        new_detection["category_id"] = find_category(track.label, labels["categories"])
        new_detection["isbbox"] = True
        new_detection["bbox"] = list(map(int, track.to_tlwh().tolist()))
        new_detection["area"] = int(new_detection["bbox"][2] * new_detection["bbox"][3])
        labels["annotations"].append(new_detection)
        annot_id += 1
    return labels


def find_category(label, categories):
    for category in categories:
        if category["name"] == label:
            return category["id"]
