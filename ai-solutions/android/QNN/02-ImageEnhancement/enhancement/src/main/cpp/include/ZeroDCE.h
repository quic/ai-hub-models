//
// Created by shivmahe on 9/5/2023.
//

#ifndef IMAGEENHANCEMENT_ZERODCE_H
#define IMAGEENHANCEMENT_ZERODCE_H

#include "Model.h"

class ZeroDCE : public Model {

public:
    ZeroDCE(){
        model_name = "ZeroDCE";
    }

    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();


};


#endif //IMAGEENHANCEMENT_ZERODCE_H
