import numpy as np
import torch

import myfunctions as myf
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
)
from yolov5.utils.torch_utils import select_device

names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_device():
    try:
        device = select_device(0)
    except Exception:
        print("Failed to load GPU")
        device = select_device("cpu")
    return device


class Yolov5:
    def __init__(self, model_path):
        self.device = load_device()
        self.model = self.load_model(model_path)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(imgsz=[640, 640], s=self.stride)

    def load_model(self, weights):
        return attempt_load(weights, map_location=self.device)

    def model_warmup(self):
        self.model(
            torch.zeros(1, 3, *self.imgsz)
            .to(self.device)
            .type_as(next(self.model.parameters()))
        )

    def inference(self, frame_number, img0, half=False):
        objects = []
        img = letterbox(img0, self.imgsz, stride=self.stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]
        pred = self.model(img)[0]
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        pred = non_max_suppression(
            pred,
            conf_thres=0.25,
            iou_thres=0.45,
            classes=None,
            agnostic=False,
            max_det=1000,
        )
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                if conf.item() < 0.5:
                    continue
                xywh = [0, 0, 0, 0]
                xywh[0] = int(xyxy[0].item())
                xywh[1] = int(xyxy[1].item())
                xywh[2] = int(xyxy[2].item() - xyxy[0].item())
                xywh[3] = int(xyxy[3].item() - xyxy[1].item())
                objects.append(
                    myf.structure_for_object(
                        frame_number,
                        img0,
                        tuple(xywh),
                        names[int(cls.item())],
                        conf.item(),
                    )
                )

        return objects
