# -*- mode: python -*-
# =============================================================================
# @@-COPYRIGHT-START-@@
#
# Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# @@-COPYRIGHT-END-@@
# =============================================================================
class SSDLITE:
    def __init__(self):
        self.mean = [127,127,127]
        self.stddev = 128
        self.class_name = ["BACKGROUND",
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
        self.label2class={str(i):x for i,x in enumerate(self.class_name)}
        self.colors = np.random.uniform(0, 255, size=(len(list(self.label2class.values())), 3))
        
    def preprocessing(self,img):
        img = cv2.resize(img, (300,300))
        img = img - self.mean
        img = img/self.stddev
        return img
        
    def postProcessinghelper(self, scores,boxes,original_image_path):
        height,width,_=original_image.shape
        prob_threshold = 0.2
        # this version of nms is slower on GPU, so we move data to CPU.
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.shape[1]):
            
            probs = scores[:, class_index]
            
            mask = probs > prob_threshold
            probs = probs[mask]
            
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            subset_boxes = torch.from_numpy(subset_boxes)
            probs = torch.from_numpy(probs)
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = nms(box_probs, None,
                                      score_threshold=prob_threshold,
                                      iou_threshold=0.2,
                                      sigma=0.2,
                                      top_k=-1,
                                      candidate_size=200)
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
                original_image = cv2.rectangle(original_image,(x, y), (x_plus_w,y_plus_h),colors[class_index],2)
                original_image=cv2.putText(original_image, label,(int(picked_box_probs[i, 0])+9, int(picked_box_probs[i, 1])+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 40, 255),2)  # line type
            picked_box_probs = []
            picked_labels = []
        return original_image

    def postprocessing(self,img, inf_result):
        print("dona")
        # scores = np.fromfile(result_path+'/942.raw', dtype="float32")
        scores = inf_result[:12000].copy().reshape((3000,4))
        # boxes=np.fromfile(result_path+'/993.raw', dtype="float32")
        boxes= inf_result[12000:].copy().reshape((3000,21))
        return postProcessinghelper(scores,boxes,img)
                