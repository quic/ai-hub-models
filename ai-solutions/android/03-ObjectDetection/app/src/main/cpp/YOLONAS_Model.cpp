// -*- mode: cpp -*-
// =============================================================================
// @@-COPYRIGHT-START-@@
//
// Copyright (c) 2023 of Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// @@-COPYRIGHT-END-@@
// =============================================================================
//
// Created by gsanjeev on 8/30/2023.
//

#include "YOLONAS_Model.h"

void YOLONAS_Model::preprocess(std::vector<float32_t> &dest_buffer, cv::Mat &img, std::vector<int> dims)
{
    LOGI("YOLONAS_Model preprocess");
    cv::Mat img320;
    //Resize and get the size from model itself (320x320 for YOLONAS)
    cv::resize(img,img320,cv::Size(dims[2],dims[1]),cv::INTER_LINEAR);

    float inputScale = 0.00392156862745f;    //normalization value, this is 1/255

    float * accumulator = reinterpret_cast<float *> (&dest_buffer[0]);

    //opencv read in BGRA by default
    cvtColor(img320, img320, CV_BGRA2BGR);
    //LOGI("num of channels: %d",img320.channels());
    int lim = img320.rows*img320.cols*3;
    for(int idx = 0; idx<lim; idx++)
        accumulator[idx]= img320.data[idx]*inputScale;

}

void YOLONAS_Model::postprocess(int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, std::vector<float32_t> &BBout_boxcoords, std::vector<float32_t> &BBout_class, float milli_time) {
    LOGI("YOLONAS_Model postprocess");
    std::vector<BoxCornerEncoding> Boxlist;
    std::vector<std::string> Classlist;

    //Post Processing
    for(int i =0;i<(2100);i++)  //TODO change value of 2100 to soft value
    {
        int start = i*80;
        int end = (i+1)*80;

        auto it = max_element (BBout_class.begin()+start, BBout_class.begin()+end);
        int index = distance(BBout_class.begin()+start, it);

        std::string classname = classnamemapping[index];
        if(*it>=0.5 )
        {
            int x1 = BBout_boxcoords[i * 4 + 0];
            int y1 = BBout_boxcoords[i * 4 + 1];
            int x2 = BBout_boxcoords[i * 4 + 2];
            int y2 = BBout_boxcoords[i * 4 + 3];
            Boxlist.push_back(BoxCornerEncoding(x1, y1, x2, y2,*it,classname));
        }
    }

    //LOGI("Boxlist size:: %d",Boxlist.size());
    std::vector<BoxCornerEncoding> reslist = NonMaxSuppression(Boxlist,0.20);
    //LOGI("reslist ssize %d", reslist.size());

    numberofobj = reslist.size();
    float ratio_2 = orig_width/320.0f;
    float ratio_1 = orig_height/320.0f;
    //LOGI("ratio1 %f :: ratio_2 %f",ratio_1,ratio_2);

    for(int k=0;k<numberofobj;k++) {
        float top,bottom,left,right;
        left = reslist[k].y1 * ratio_1;   //y1
        right = reslist[k].y2 * ratio_1;  //y2

        bottom = reslist[k].x1 * ratio_2;  //x1
        top = reslist[k].x2 * ratio_2;   //x2

        //LOGI("yolonas box postprocess: orig_width=%d orig_height=%d",orig_width, orig_height);
        //LOGI("yolonas box postprocess: top:x2=%f bottom:x1=%f left:y1=%f right:y2=%f",top,bottom,left,right);

        std::vector<float> singleboxcoords{top, bottom, left, right, milli_time};
        BB_coords.push_back(singleboxcoords);
        BB_names.push_back(reslist[k].objlabel);
    }
}

void YOLONAS_Model::msg()
{
    LOGI("YOLONAS_Model Class msg model_name = %d",model_name);
}