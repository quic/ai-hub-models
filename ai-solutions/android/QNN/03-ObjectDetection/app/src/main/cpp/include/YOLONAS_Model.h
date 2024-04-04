//
// Created by gsanjeev on 8/30/2023.
//

#ifndef APP_YOLONAS_MODEL_H
#define APP_YOLONAS_MODEL_H

#include "Model.h"

#include <map>

class BoxCornerEncoding {

public:
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    std::string objlabel;

    BoxCornerEncoding(int a, int b, int c, int d,int sc, std::string name="person")
    {
        x1 = a;
        y1 = b;
        x2 = c;
        y2 = d;
        score = sc;
        objlabel = name;
    }
};


class YOLONAS_Model: public Model{


    std::map<int, std::string> classnamemapping =
            {
                    {0, "person"},{ 1, "bicycle"},{ 2, "car"},{ 3, "motorcycle"},{ 4, "airplane"},{ 5, "bus"},{
                     6, "train"},{ 7, "truck"},{ 8, "boat"},{ 9, "traffic"},{ 10, "fire"},{ 11, "stop"},{ 12, "parking"},{
                     13, "bench"},{ 14, "bird"},{ 15, "cat"},{ 16, "dog"},{ 17, "horse"},{ 18, "sheep"},{ 19, "cow"},{
                     20, "elephant"},{ 21, "bear"},{ 22, "zebra"},{ 23, "giraffe"},{ 24, "backpack"},{ 25, "umbrella"},{
                     26, "handbag"},{ 27, "tie"},{ 28, "suitcase"},{ 29, "frisbee"},{ 30, "skis"},{ 31, "snowboard"},{
                     32, "sports"},{ 33, "kite"},{ 34, "baseball"},{ 35, "baseball"},{ 36, "skateboard"},{ 37, "surfboard"},{
                     38, "tennis"},{ 39, "bottle"},{ 40, "wine"},{ 41, "cup"},{ 42, "fork"},{ 43, "knife"},{ 44, "spoon"},{
                     45, "bowl"},{ 46, "banana"},{ 47, "apple"},{ 48, "sandwich"},{ 49, "orange"},{ 50, "broccoli"},{
                     51, "carrot"},{ 52, "hot"},{ 53, "pizza"},{ 54, "donut"},{ 55, "cake"},{ 56, "chair"},{ 57, "couch"},{
                     58, "potted"},{ 59, "bed"},{ 60, "dining"},{ 61, "toilet"},{ 62, "tv"},{ 63, "laptop"},{ 64, "mouse"},{
                     65, "remote"},{ 66, "keyboard"},{ 67, "cell"},{ 68, "microwave"},{ 69, "oven"},{ 70, "toaster"},{
                     71, "sink"},{ 72, "refrigerator"},{ 73, "book"},{ 74, "clock"},{ 75, "vase"},{ 76, "scissors"},{
                     77, "teddy"},{ 78, "hair"},{ 79, "toothbrush"}
            };

public:

    YOLONAS_Model()
    {
        model_name=YOLONAS;
    }

    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, std::vector<float32_t> &BBout_boxcoords, std::vector<float32_t> &BBout_class, float milli_time);
    void msg();

    inline float ComputeIntersectionOverUnion(const BoxCornerEncoding &box_i,const BoxCornerEncoding &box_j)
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

    std::vector<BoxCornerEncoding> NonMaxSuppression(std::vector<BoxCornerEncoding> boxes,
                                                     const float iou_threshold)
    {

        if (boxes.size()==0) {
            return boxes;
        }

        std::sort(boxes.begin(), boxes.end(), [] (const BoxCornerEncoding& left, const BoxCornerEncoding& right) {
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

        std::vector<BoxCornerEncoding> ret;
        for (unsigned int i = 0; i < boxes.size(); i++) {
            if (!flag[i])
                ret.push_back(boxes[i]);
        }

        return ret;
    }

};


#endif //APP_YOLONAS_MODEL_H
