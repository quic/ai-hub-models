//
// Created by gsanjeev on 8/30/2023.
//

#include "../include/YOLONAS_Model.h"

void YOLONAS_Model::preprocess(cv::Mat &img, std::vector<int> dims)
{
    LOGI("YOLONAS_Model preprocess");

    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,img,cv::Size(dims[1],dims[0]), 0, 0, cv::INTER_LINEAR); //Resizing based on input
    LOGI("inputimage SIZE width::%d height::%d channels::%d",img.cols, img.rows, img.channels());

    float inputScale = 0.00392156862745f;    //normalization value, this is 1/255

    //opencv read in BGRA by default
    cvtColor(img, img, CV_BGRA2BGR);
    img.convertTo(img,CV_32FC3,inputScale);
    LOGI("num of channels: %d",img.channels());

    float* temp = reinterpret_cast<float *>(img.data);
    for(int i=0;i<10;i++)
    {
        __android_log_print(ANDROID_LOG_ERROR, "QNN ", "input buffer[%d] = %f\n",i,temp[i]);
    }
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

        // 0.61 0.34
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

//    LOGI("YOLONAS_Model postprocess: Completed final loop");
}

void YOLONAS_Model::msg()
{
    LOGI("YOLONAS_Model Class msg model_name = %d",model_name);
}