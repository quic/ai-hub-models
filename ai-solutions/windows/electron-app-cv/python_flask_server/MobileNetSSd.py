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

# from torch import Tensor, clamp, max, min, from_numpy, cat
import torch

from utils import draw_box

class_name = ["BACKGROUND",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"]
label2class={str(i):x for i,x in enumerate(class_name)}
colors = np.random.uniform(0, 255, size=(len(list(label2class.values())), 3))

class MobileNetSSD:
    def __init__(self):
        self.mean = [123,117,104]
        
        
    def area_of(self, left_top, right_bottom) -> torch.Tensor:
        """Compute the areas of rectangles given two corners.

        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.

        Returns:
            area (N): return the area.
        """
        hw = torch.clamp(right_bottom - left_top, min=0.0)
        return hw[..., 0] * hw[..., 1]




    def iou_of(self, boxes0, boxes1, eps=1e-5):
        """Return intersection-over-union (Jaccard index) of boxes.

        Args:
            boxes0 (N, 4): ground truth boxes.
            boxes1 (N or 1, 4): predicted boxes.
            eps: a small number to avoid 0 as denominator.
        Returns:
            iou (N): IoU values.
        """
        overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)




    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        """

        Args:
            box_scores (N, 5): boxes in corner-form and probabilities.
            iou_threshold: intersection over union threshold.
            top_k: keep top_k results. If k <= 0, keep all the results.
            candidate_size: only consider the candidates with the highest scores.
        Returns:
             picked: a list of indexes of the kept boxes
        """
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        print(boxes.shape)
        _, indexes = scores.sort(descending=True)
        indexes = indexes[:candidate_size]
        while len(indexes) > 0:
            current = indexes[0]
            picked.append(current.item())
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[1:]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                current_box.unsqueeze(0),
            )
            indexes = indexes[iou <= iou_threshold]

        return box_scores[picked, :]



    def nms(self, box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
            sigma=0.5, top_k=-1, candidate_size=200):
        if nms_method == "soft":
            return self.soft_nms(box_scores, score_threshold, sigma, top_k)
        else:
            return self.hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)
        
    def preprocessing(self,img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320,320))
        # print("image shape: ",img.shape)
        # print("before->",img[0][0][:])
        # img = img - self.mean
        # print("end->",img[0][0][:])
        # img = img* 1.070312500000
        return img

    def postProcessinghelper(self, scores, boxes, original_image):
        height,width,_=original_image.shape
        # print("originam_image shaoe",original_image.shape)
        prob_threshold = 0.4
        picked_box_probs = []
        picked_labels = []
        # print("scores shape: ",scores.shape)
        for class_index in range(1, scores.shape[1]):
            
            probs = scores[:, class_index]
            print("highest: ",np.max(probs))
            # print("probs",probs)
            mask = probs > prob_threshold
            probs = probs[mask]
            
            if probs.shape[0] == 0:
                # print("Continue")
                continue
            subset_boxes = boxes[mask, :]
            subset_boxes = torch.from_numpy(subset_boxes)
            probs = torch.from_numpy(probs)
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = self.nms(box_probs, None,
                                      score_threshold=prob_threshold,
                                      iou_threshold=0.2,
                                      sigma=0.2,
                                      top_k=-1,
                                      candidate_size=200)
            # print("box_prods is calculated")
            picked_box_probs.extend([box_probs])
            picked_labels.extend([class_index] * box_probs.size(0))
            picked_box_probs = torch.cat(picked_box_probs)
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height
            label = class_name[picked_labels[0]]
            
            for i in range(0,len(picked_box_probs)):
                x,y=int(picked_box_probs[i, 0].numpy()),int(picked_box_probs[i, 1].numpy())
                x_plus_w,y_plus_h=int(picked_box_probs[i, 2].numpy()),int(picked_box_probs[i, 3].numpy())
                # print("cords: ", x, "::", y,"::", x_plus_w, "::", y_plus_h)
                # print(picked_box_probs)
                # original_image = cv2.rectangle(original_image,(x, y), (x_plus_w,y_plus_h),colors[class_index],2)
                # original_image=cv2.putText(original_image, label,(int(picked_box_probs[i, 0])+9, int(picked_box_probs[i, 1])-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 40, 255),2)  # line 
                draw_box(original_image,[x,y,x_plus_w,y_plus_h],label,picked_box_probs[i,4].tolist(),colors[class_index])
               
            picked_box_probs = []
            picked_labels = []
        return original_image

    def postprocessing(self,img, inf_result):
        # print("dona")
        # scores = np.fromfile(result_path+'/942.raw', dtype="float32")
        scores = inf_result[:67914].copy().reshape((3234,21))
        # boxes=np.fromfile(result_path+'/993.raw', dtype="float32")
        boxes= inf_result[67914:].copy().reshape((3234,4))
        return self.postProcessinghelper(scores,boxes,cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
                