//
// Created by shubgoya on 8/2/2023.
//

#ifndef SUPERRESOLUTION_XLSR_H
#define SUPERRESOLUTION_XLSR_H

#include "Model.h"

class XLSR: public Model {

public:
    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};


#endif //SUPERRESOLUTION_XLSR_H