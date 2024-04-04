//
// Created by shubgoya on 8/2/2023.
//

#ifndef SUPERRESOLUTION_QUICKSRNETMEDIUM_H
#define SUPERRESOLUTION_QUICKSRNETMEDIUM_H

#include "Model.h"

class QuickSRNetMedium: public Model {

public:
    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};


#endif //SUPERRESOLUTION_QUICKSRNETMEDIUM_H