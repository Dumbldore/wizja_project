# (Length, width, height) in meters

default_objects_sizes = {
    "person": (0.3, 0.6, 1.8),
    "people": (0.3, 0.6, 1.8),
    "wheelchair": (1.1, 0.7, 1.5),
    "bicycle": (1.72, 0.76, 1.5),
    "car": (4.5, 1.8, 1.6),
    "police": (4.5, 1.8, 1.6),
    "motorcycle": (1.72, 0.76, 1.5),
    "bus": (12, 2.8, 3.5),
    "truck": (14, 2.8, 4.5),
    "van": (6.75, 2.37, 3),
}

min_objects_sizes = {
    "person": (0.1, 0.2, 1),
    "people": (0.1, 0.2, 1),
    "wheelchair": (0.6, 0.3, 0.6),
    "bicycle": (0.7, 0.3, 0.6),
    "car": (2, 1, 1),
    "police": (2, 1, 1),
    "motorcycle": (0.7, 0.3, 0.6),
    "bus": (4, 1.5, 1.5),
    "truck": (4, 1.5, 1.5),
    "van": (3, 1.5, 1.3),
}

max_objects_sizes = {
    "person": (1.5, 2, 3),
    "people": (1.5, 2, 5),
    "wheelchair": (2.5, 2.5, 3),
    "bicycle": (3, 2, 3),
    "car": (8, 3, 4),
    "police": (8, 3, 4),
    "motorcycle": (5, 3, 3),
    "bus": (30, 6, 6),
    "truck": (40, 6, 8),
    "van": (12, 5, 5),
}

vehicles = ("bicycle", "car", "police", "motorcycle", "bus", "truck", "van")
vertical_objects = ("person", "wheelchair")
