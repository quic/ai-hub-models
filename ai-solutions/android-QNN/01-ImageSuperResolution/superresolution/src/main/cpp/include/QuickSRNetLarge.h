//
// Created by shubgoya on 8/2/2023.
//

#ifndef SUPERRESOLUTION_QuickSRNetLarge_H
#define SUPERRESOLUTION_QuickSRNetLarge_H

#include "Model.h"

class QuickSRNetLarge: public Model {

public:
    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};


#endif //SUPERRESOLUTION_QuickSRNetLarge_H