//
// Created by gsanjeev on 9/11/2023.
//

#include "../include/SSDMobileNetV2_Model.h"


void SSDMobileNetV2_Model::preprocess(cv::Mat &img, std::vector<int> dims)
{
    LOGI("SSDMobileNetV2_Model preprocess");

    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,img,cv::Size(dims[1],dims[0]), 0, 0, cv::INTER_LINEAR); //Resizing based on input
    LOGI("inputimage SIZE width::%d height::%d",img.cols, img.rows);

//    float inputScale = 0.00392156862745f;    //normalization value, this is 1/255
    float inputScale = 1.070312500000f;    //normalization value, this is 1/255

    //opencv read in BGRA by default
    cvtColor(img, img, CV_BGRA2RGB);
    img.convertTo(img,CV_32FC3,inputScale);
    LOGI("num of channels: %d",img.channels());

    float* temp = reinterpret_cast<float *>(img.data);
    for(int i=0;i<10;i++)
    {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "input buffer[%d] = %f\n",i,temp[i]);
    }
}

void SSDMobileNetV2_Model::postprocess(int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, std::vector<float32_t> &BBout_boxcoords, std::vector<float32_t> &BBout_class, float milli_time) {
    LOGI("SSDMobileNetV2_Model postprocess");
    std::vector<SSDMobileNetV2BoxCornerEncoding> Boxlist;
    std::vector<std::string> Classlist;

    //sanjeev temp sanity check for sometimes stability issue in SSDMobileNetV2 Model
    if (BBout_boxcoords.size() == 0)
    {
        numberofobj=-1;
        LOGE("sanjeev BBout_boxcoords is zero. Returning Error..");
        return;
    }

    //Post Processing
    for(int i =1;i<(21);i++) // [21 classes supported by SSDMobileNetV2]
    {

        int row_index;
        float max_element;

        std::string classname = classnamemapping[i];

        for (int j=i; j<(67914); j+=21) // [67914 = 21 (no of classes) x 3234 (total boxes output by model)]
        {
            if (BBout_class[j] > 0.4)
            {
                max_element = BBout_class[j];
                row_index = j/21;

                float x1 = BBout_boxcoords[row_index * 4 + 0];
                float y1 = BBout_boxcoords[row_index * 4 + 1];
                float x2 = BBout_boxcoords[row_index * 4 + 2];
                float y2 = BBout_boxcoords[row_index * 4 + 3];

                Boxlist.push_back(SSDMobileNetV2BoxCornerEncoding(x1, y1, x2, y2,max_element,classname));
            }
        }

    }

    LOGI("Boxlist size:: %d",Boxlist.size());
    std::vector<SSDMobileNetV2BoxCornerEncoding> reslist = NonMaxSuppression(Boxlist,0.20);
    LOGI("reslist size %d", reslist.size());

    numberofobj = reslist.size();

    //LOGI("numberofobj detected = %d", numberofobj);

    float ratio_2 = orig_width;
    float ratio_1 = orig_height;

    //LOGI("ratio1 %f :: ratio_2 %f",ratio_1,ratio_2);

    for(int k=0;k<numberofobj;k++) {
        float top,bottom,left,right;
        left = reslist[k].y1 * ratio_1;   //y1
        right = reslist[k].y2 * ratio_1;  //y2

        bottom = reslist[k].x1 * ratio_2;  //x1
        top = reslist[k].x2 * ratio_2;   //x2

        //LOGI("SSDMobileNetV2 box postprocess: orig_width=%d orig_height=%d",orig_width, orig_height);
        //LOGI("SSDMobileNetV2 box postprocess: top:x2=%f bottom:x1=%f left:y1=%f right:y2=%f",top,bottom,left,right);

        std::vector<float> singleboxcoords{top, bottom, left, right, milli_time};
        BB_coords.push_back(singleboxcoords);
        BB_names.push_back(reslist[k].objlabel);
    }

}

void SSDMobileNetV2_Model::msg()
{
    LOGI("SSDMobileNetV2_Model Class msg model_name = %d", model_name);
}