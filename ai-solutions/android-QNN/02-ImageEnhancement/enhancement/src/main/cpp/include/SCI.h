//
// Created by shivmahe on 9/5/2023.
//

#ifndef IMAGEENHANCEMENT_SCI_H
#define IMAGEENHANCEMENT_SCI_H

#include "Model.h"

class SCI : public Model {

public:
    SCI(){
        model_name = "SCI";
    }

    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();


};


#endif //IMAGEENHANCEMENT_SCI_H
