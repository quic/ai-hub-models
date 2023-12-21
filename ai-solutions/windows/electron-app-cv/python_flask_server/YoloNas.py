# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
import cv2
import numpy as np

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

class YoloNAS:
        
    def preprocessing(self,img):
        img = cv2.resize(img, (320,320))
        img = img/255
        return img
        
    def postprocessing(self, image_data, inf_result):
        # print("step-2")
        print(inf_result)
        raw_scores = inf_result[:168000].copy() #.reshape(YOLONAS_MODEL_CLASSES_OUTPUT_SIZE)
        output_909_reshape = raw_scores.reshape(2100,80)
        # print("step-1")
        output = output_909_reshape
        raw_boxes = inf_result[168000:].copy() #.reshape(YOLONAS_MODEL_BOXES_OUTPUT_SIZE)
        output_917_reshape = raw_boxes.reshape(2100,4)
        
        # print("step0")
        boxes = []
        scores = []
        class_ids = []
        original_image = np.array(image_data, np.float32)
        ratio_1 = original_image.shape[0]/320
        ratio_2 = original_image.shape[1]/320
        
        # print("step 1")
        for i in range(0, output.shape[0]):
            classes_scores = output[i]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.05:
                x = round(output_917_reshape[i][0]) ; y = round(output_917_reshape[i][1]); 
                w = round(output_917_reshape[i][2]) ; h = round(output_917_reshape[i][3]);
        
                x1, y1 = x, y
                x2, y2 = w, h
                box = [x1, y1, x2, y2]
                boxes.append(box)
                scores.append(float(maxScore))
                class_ids.append(maxClassIndex)
        
        # print("step2")
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.40, 0.5, 0.5) #32b CPU

        # print("result_boxes :: ",result_boxes)
        detections = []
        img = np.array(image_data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]

            print("-----------")
            detection = {
                'class_id': class_ids[index],
                'class_name': label2class[str(class_ids[index])],
                'confidence': scores[index],
                'box': box
                 }
            detections.append(detection)
            img = self.draw_bounding_box(img, class_ids[index], scores[index], int(box[0]*ratio_2), int(box[1]*ratio_1), int(box[2]*ratio_2), int(box[3]*ratio_1))
            
        return img
        
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f'{label2class[str(class_id)]} ({confidence:.2f})'
        color = colors[class_id]
        draw_box(img,[x,y,x_plus_w,y_plus_h],label,confidence,color)
        return img
    
