#!/usr/bin/env python3
"""Object Detection"""
import numpy as np


class Yolo:
    """Class of Yolo"""
    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip()
                                for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process Outputs"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        image_h, image_w = image_size

        for i in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = outputs.shape[:3]
            t_x = outputs[..., 0]
            t_y = outputs[..., 1]
            t_w = outputs[..., 2]
            t_h = outputs[..., 3]
            box_confidence = 1 / (1 + np.exp(-outputs[..., 4]))
            box_class_prob = 1 / (1 + np.exp(-outputs[..., 5:]))

            c_x = np.arange(grid_w).reshape(1, grid_w, 1)
            c_y = np.arange(grid_h).reshape(grid_h, 1, 1)

            bx = (1 / (1 + np.exp(-t_x)) + c_x) / grid_w
            by = (1 / (1 + np.exp(-t_y)) + c_y) / grid_h

            bw = (self.anchors[i, :, 0] * np.exp(t_w)) / self.model.input.shape[1]
            bh = (self.anchors[i, :, 1] * np.exp(t_h)) / self.model.input.shape[2]

            x1 = (bx - (bw / 2)) * image_w
            y1 = (by - (bh / 2)) * image_h
            x2 = (bx + (bw / 2)) * image_w
            y2 = (by + (bh / 2)) * image_h

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(box_confidence[..., np.newaxis])
            box_class_probs.append(box_class_prob)

        return boxes, box_confidences, box_class_probs
