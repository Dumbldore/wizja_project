import numpy as np


def structure_for_object(current_frame, frame, dbox, dclass, confidence):
    """
    This function creates dict for each object before those are append
    to the actual list of objects. Nothing fancy happens here: based
    on boudning box (dbox) and class (dclass) of an object box, class,
    unique set of colors, tracker and unique ID for each object are
    organised and stored.
    """
    obj = {}
    obj["Box"] = dbox
    obj["Class"] = dclass
    obj["Colors"] = (
        np.random.randint(0, 255),
        np.random.randint(0, 255),
        np.random.randint(0, 255),
    )
    obj["ID"] = "#" + str(np.random.randint(0, 10000))
    obj["Frame"] = current_frame
    obj["Confidence"] = confidence

    return obj


def update_objects_trackers_kalman(frame, tracks):
    objects = []
    for track in tracks:
        if track.cv_tracker == None:
            continue
        state, update = track.cv_tracker.update(frame)
        if state is False:
            continue
        else:
            obj = {}
            int_update = [int(x) for x in update]
            checked_update = [entry < 0 for entry in int_update]
            if True in checked_update:
                for idy, entry in enumerate(checked_update):
                    if entry is True:
                        int_update[idy] = 0

            obj["Box"] = tuple(int_update)
            obj["Class"] = track.label
            obj["Confidence"] = 0.4
            objects.append(obj)
    return objects
