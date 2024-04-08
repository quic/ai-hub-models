//
// Created by shivmahe on 9/5/2023.
//

#ifndef IMAGEENHANCEMENT_MBLLEN_H
#define IMAGEENHANCEMENT_MBLLEN_H

#include "Model.h"

class MBLLEN: public Model {

public:
    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();

};

#endif //IMAGEENHANCEMENT_MBLLEN_H
