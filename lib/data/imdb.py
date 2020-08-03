# --------------------------------------------------------
# Fast R-CNN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import numpy as np
import torch
from ensemble_boxes import *


def detect1Image(im0, imgsz, model, device, conf_thres, iou_thres, aug = False):
    img = letterbox(im0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)


    img = torch.from_numpy(img).to(device)
    img =  img.float()  # uint8 to fp16/32
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=aug)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    boxes = []
    scores = []
    for i, det in enumerate(pred):  # detections per image
        # save_path = 'draw/' + image_id + '.jpg'
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                scores.append(conf)

    return np.array(boxes), np.array(scores)
