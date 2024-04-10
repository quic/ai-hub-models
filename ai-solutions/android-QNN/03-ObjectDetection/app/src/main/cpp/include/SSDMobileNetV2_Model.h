//
// Created by gsanjeev on 9/11/2023.
//

#ifndef APP_SSDMOBILENETV2_MODEL_H
#define APP_SSDMOBILENETV2_MODEL_H

#include "Model.h"

#include <map>

class SSDMobileNetV2BoxCornerEncoding {

public:
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    std::string objlabel;

    SSDMobileNetV2BoxCornerEncoding(float a, float b, float c, float d,float sc, std::string name="person")
    {
        x1 = a;
        y1 = b;
        x2 = c;
        y2 = d;
        score = sc;
        objlabel = name;
    }
};



class SSDMobileNetV2_Model: public Model{

    std::map<int, std::string> classnamemapping =
            {
                    {0, "BACKGROUND"},{ 1, "aeroplane"},{ 2, "bicycle"},{ 3, "bird"},{ 4, "boat"},{ 5, "bottle"},{
                     6, "bus"},{ 7, "car"},{ 8, "cat"},{ 9, "chair"},{ 10, "cow"},{ 11, "diningtable"},{ 12, "dog"},{
                     13, "horse"},{ 14, "motorbike"},{ 15, "person"},{ 16, "pottedplant"},{ 17, "sheep"},{ 18, "sofa"},{ 19, "train"},{
                     20, "tvmonitor"}
            };

public:

    SSDMobileNetV2_Model()
    {
        model_name=SSDMobilenetV2;
    }

    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, std::vector<float32_t> &BBout_boxcoords, std::vector<float32_t> &BBout_class, float milli_time);
    void msg();

    inline float ComputeIntersectionOverUnion(const SSDMobileNetV2BoxCornerEncoding &box_i,const SSDMobileNetV2BoxCornerEncoding &box_j)
    {
        const float box_i_y_min = std::min<float>(box_i.y1, box_i.y2);
        const float box_i_y_max = std::max<float>(box_i.y1, box_i.y2);
        const float box_i_x_min = std::min<float>(box_i.x1, box_i.x2);
        const float box_i_x_max = std::max<float>(box_i.x1, box_i.x2);
        const float box_j_y_min = std::min<float>(box_j.y1, box_j.y2);
        const float box_j_y_max = std::max<float>(box_j.y1, box_j.y2);
        const float box_j_x_min = std::min<float>(box_j.x1, box_j.x2);
        const float box_j_x_max = std::max<float>(box_j.x1, box_j.x2);

        const float area_i =
                (box_i_y_max - box_i_y_min) * (box_i_x_max - box_i_x_min);
        const float area_j =
                (box_j_y_max - box_j_y_min) * (box_j_x_max - box_j_x_min);
        if (area_i <= 0 || area_j <= 0) return 0.0;
        const float intersection_ymax = std::min<float>(box_i_y_max, box_j_y_max);
        const float intersection_xmax = std::min<float>(box_i_x_max, box_j_x_max);
        const float intersection_ymin = std::max<float>(box_i_y_min, box_j_y_min);
        const float intersection_xmin = std::max<float>(box_i_x_min, box_j_x_min);
        const float intersection_area =
                std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
                std::max<float>(intersection_xmax - intersection_xmin, 0.0);
        return intersection_area / (area_i + area_j - intersection_area);
    }

    std::vector<SSDMobileNetV2BoxCornerEncoding> NonMaxSuppression(std::vector<SSDMobileNetV2BoxCornerEncoding> boxes,
                                                            const float iou_threshold)
    {

        if (boxes.size()==0) {
            return boxes;
        }

        std::sort(boxes.begin(), boxes.end(), [] (const SSDMobileNetV2BoxCornerEncoding& left, const SSDMobileNetV2BoxCornerEncoding& right) {
            if (left.score > right.score) {
                return true;
            } else {
                return false;
            }
        });


        std::vector<bool> flag(boxes.size(), false);
        for (unsigned int i = 0; i < boxes.size(); i++) {
            if (flag[i]) {
                continue;
            }

            for (unsigned int j = i + 1; j < boxes.size(); j++) {
                if (ComputeIntersectionOverUnion(boxes[i],boxes[j]) > iou_threshold) {
                    flag[j] = true;
                }
            }
        }

        std::vector<SSDMobileNetV2BoxCornerEncoding> ret;
        for (unsigned int i = 0; i < boxes.size(); i++) {
            if (!flag[i])
                ret.push_back(boxes[i]);
        }

        return ret;
    }

};


#endif //APP_SSDMOBILENETV2_MODEL_H
