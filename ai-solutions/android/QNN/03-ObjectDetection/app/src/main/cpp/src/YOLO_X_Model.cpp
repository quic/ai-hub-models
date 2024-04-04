//
// Created by gsanjeev on 9/20/2023.
//

#include "../include/YOLO_X_Model.h"


void YOLO_X_Model::preprocess(cv::Mat &img, std::vector<int> dims)
{
    LOGI("YOLO_X_Model preprocess");

    //dims is of size [batchsize(1), height, width, channels(3)]
    cv::resize(img,img,cv::Size(dims[1],dims[0]), 0, 0, cv::INTER_LINEAR); //Resizing based on input
    LOGI("inputimage SIZE width::%d height::%d",img.cols, img.rows);

    float inputScale = 0.998046875000f;    //normalization value, this is 1/255

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

void YOLO_X_Model::postprocess(int orig_width, int orig_height, int &numberofobj, std::vector<std::vector<float>> &BB_coords, std::vector<std::string> &BB_names, std::vector<float32_t> &BBout_boxcoords, std::vector<float32_t> &BBout_class, float milli_time) {

    //YoloX Model has only single tensor output (BBout_class) that includes classes, boxes as well as score

    LOGI("YOLO_X_Model postprocess");
    std::vector<YOLO_X_BoxCornerEncoding> Boxlist;
    std::vector<std::string> Classlist;

    std::vector<int> img_size = {640, 640};
    bool p6=false;

    struct Point {
        float32_t x,y;
    };

    std::vector<Point> grids;
    std::vector<float32_t> expanded_strides;
    std::vector<int> strides = !p6 ? std::vector<int>{8, 16, 32} : std::vector<int>{8, 16, 32, 64};

    std::vector<int> hsizes, wsizes;
    for (const int stride : strides) {
        hsizes.push_back(img_size[0] / stride);
        wsizes.push_back(img_size[1] / stride);
    }

    for (size_t i=0; i<hsizes.size(); ++i) {

        int hsize = hsizes[i];
        int wsize = wsizes[i];
        int stride = strides[i];

        for (int y=0; y<hsize; ++y) {

            for (int x=0; x<wsize; ++x) {

                Point point{static_cast<float32_t>(x), static_cast<float32_t>(y)};
                grids.push_back(point);
                expanded_strides.push_back(stride);
            }
        }
    }

    for (size_t i=0; i<8400; i++) {

        BBout_class[i*85+0] = (BBout_class[i*85+0] + grids[i].x) * expanded_strides[i];
        BBout_class[i*85+1] = (BBout_class[i*85+1] + grids[i].y) * expanded_strides[i];
        BBout_class[i*85+2] = std::exp(BBout_class[i*85+2]) * expanded_strides[i];
        BBout_class[i*85+3] = std::exp(BBout_class[i*85+3]) * expanded_strides[i];

    }

    for(int i =0;i<(8400);i++)  // Total tensor output rows
    {
        int start = i*85+5; // each row contains classes from indexes 5 to 85
        int end = (i+1)*85;

        float score = BBout_class[start-1]; // each row contains score at index 4

        if(score>=0.2 )
        {

            auto it = max_element (BBout_class.begin()+start, BBout_class.begin()+end);

            int index = distance(BBout_class.begin()+start, it);

            std::string classname = classnamemapping[index];

            // each row contains box co-ordinates from index 0 to 3
            float x1 = BBout_class[start-5];
            float y1 = BBout_class[start-4];
            float x2 = BBout_class[start-3];
            float y2 = BBout_class[start-2];
            Boxlist.push_back(YOLO_X_BoxCornerEncoding(x1, y1, x2, y2,score,classname));
        }
    }

    std::vector<YOLO_X_BoxCornerEncoding> reslist_temp = Yolo_X_Rescale_boxes(Boxlist,orig_width,orig_height);

    std::vector<YOLO_X_BoxCornerEncoding> reslist = NonMaxSuppression(reslist_temp,0.20);
    //LOGI("reslist ssize %d", reslist.size());


    numberofobj = reslist.size();

    float ratio_2 = orig_width/640.0f;
    float ratio_1 = orig_height/640.0f;
    //LOGI("ratio1 %f :: ratio_2 %f",ratio_1,ratio_2);

    for(int k=0;k<numberofobj;k++) {
        float top,bottom,left,right;
        left = reslist[k].y1 * ratio_1;   //y1
        right = reslist[k].y2 * ratio_1;  //y2

        bottom = reslist[k].x1 * ratio_2;  //x1
        top = reslist[k].x2 * ratio_2;   //x2

        //LOGI("YOLO_X_Model box postprocess: orig_width=%d orig_height=%d",orig_width, orig_height);
        //LOGI("YOLO_X_Model box postprocess: top:x2=%f bottom:x1=%f left:y1=%f right:y2=%f",top,bottom,left,right);

        std::vector<float> singleboxcoords{top, bottom, left, right, milli_time};
        BB_coords.push_back(singleboxcoords);
        BB_names.push_back(reslist[k].objlabel);
    }
}

void YOLO_X_Model::msg()
{
    LOGI("YOLO_X_Model Class msg model_name = %d", model_name);
}