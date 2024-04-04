//
// Created by shubgoya on 8/2/2023.
//

#ifndef SUPERRESOLUTION_QuickSRNetSmall_H
#define SUPERRESOLUTION_QuickSRNetSmall_H

#include "Model.h"

class QuickSRNetSmall: public Model {

public:
    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};


#endif //SUPERRESOLUTION_QuickSRNetSmall_H