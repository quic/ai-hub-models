# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
from PIL import Image
import io
import os
import cv2
import numpy as np
import time
import zmq
import sys
from utils import draw_box

label2class = {'0': 'person', '1': 'bicycle', '2': 'car', '3': 'motorcycle', '4': 'airplane', '5': 'bus', 
       '6': 'train', '7': 'truck', '8': 'boat', '9': 'traffic', '10': 'fire', '11': 'stop', '12': 'parking', 
       '13': 'bench', '14': 'bird', '15': 'cat', '16': 'dog', '17': 'horse', '18': 'sheep', '19': 'cow', 
       '20': 'elephant', '21': 'bear', '22': 'zebra', '23': 'giraffe', '24': 'backpack', '25': 'umbrella', 
       '26': 'handbag', '27': 'tie', '28': 'suitcase', '29': 'frisbee', '30': 'skis', '31': 'snowboard', 
       '32': 'sports', '33': 'kite', '34': 'baseball', '35': 'baseball', '36': 'skateboard', '37': 'surfboard', 
       '38': 'tennis', '39': 'bottle', '40': 'wine', '41': 'cup', '42': 'fork', '43': 'knife', '44': 'spoon', 
       '45': 'bowl', '46': 'banana', '47': 'apple', '48': 'sandwich', '49': 'orange', '50': 'broccoli', 
       '51': 'carrot', '52': 'hot', '53': 'pizza', '54': 'donut', '55': 'cake', '56': 'chair', '57': 'couch', 
       '58': 'potted', '59': 'bed', '60': 'dining', '61': 'toilet', '62': 'tv', '63': 'laptop', '64': 'mouse', 
       '65': 'remote', '66': 'keyboard', '67': 'cell', '68': 'microwave', '69': 'oven', '70': 'toaster', 
       '71': 'sink', '72': 'refrigerator', '73': 'book', '74': 'clock', '75': 'vase', '76': 'scissors', 
       '77': 'teddy', '78': 'hair', '79': 'toothbrush'}
colors = np.random.uniform(0, 255, size=(len(list(label2class.values())), 3))

class YoloX:
    def __init__(self):
        self.input_shape = tuple(map(int,[640,640]))

        
    def preproc_helper(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def preprocessing(self,img):
        img = np.array(img, np.float32)
        img, self.ratio = self.preproc_helper(img, self.input_shape)
        img = np.transpose(img,(1,2,0))
       
        return img
        
    def draw_bounding_box(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f'{label2class[str(class_id)]}'
        color = colors[class_id]
        draw_box(img,[x,y,x_plus_w,y_plus_h],label,confidence,color)
        # img = cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        # img = cv2.putText(img, label, (x +2, y -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 2)
        # print("Label of oject:", label)
        # print("confidence", confidence)
        return img

    def demo_postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        newoutputs = np.copy(outputs)
        newoutputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        finaloutputs = np.copy(newoutputs)
        finaloutputs[..., 2:4] = np.exp(newoutputs[..., 2:4]) * expanded_strides

        return finaloutputs

    def postprocessing(self, image_data, inf_result):
        
        inf_result = inf_result.reshape((1,8400, 85))
        output = self.demo_postprocess(inf_result,self.input_shape)[0]
       
        #Initializing the lists
        boxes_updated = []
        scores_updated = []
        class_ids = []

        # Preprocessing the boxes and scores
        #format of output is first 4 is the bounding boxes, 5th one is objectness score, last 80 column is score of each classes
        boxes = output[:, :4]
        scores = output[:, 4:5] * output[:, 5:]

        #Processing of bounding boxes
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= self.ratio
        
        #For each prediction from 8400 predictions finding the results
        for i in range(0, output.shape[0]):
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(scores[i])
            if maxScore >= 0.2:
                boxes_updated.append(boxes_xyxy[i])
                # print("boxes_xyxy",boxes_xyxy[i][:])
                scores_updated.append(float(maxScore))
                class_ids.append(maxClassIndex)

        # Removing Overlapping predictions
        result_boxes = cv2.dnn.NMSBoxes(boxes_updated, scores_updated, 0.40, 0.5, 0.5) #32b CPU
        # detections = []
        img = np.array(image_data) ##int8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #For each prediction showing drawing the bounding boxes
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes_updated[index]

            detection = {
                'class_id': class_ids[index],
                'class_name': label2class[str(class_ids[index])],
                'confidence': scores_updated[index],
                'box': box
                 }
            # detections.append(detection)
            img = self.draw_bounding_box(img, class_ids[index],detection['confidence'], int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                                    
        return img
