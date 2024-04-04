//
// Created by shubgoya on 8/2/2023.
//

#ifndef SUPERRESOLUTION_SRGAN_H
#define SUPERRESOLUTION_SRGAN_H

#include "Model.h"

class SRGAN: public Model {

public:
    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};


#endif //SUPERRESOLUTION_SRGAN_H