//
// Created by shubgoya on 8/2/2023.
//

#ifndef SUPERRESOLUTION_SESR_H
#define SUPERRESOLUTION_SESR_H

#include "Model.h"

class SESR: public Model {

public:
    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};


#endif //SUPERRESOLUTION_SESR_H