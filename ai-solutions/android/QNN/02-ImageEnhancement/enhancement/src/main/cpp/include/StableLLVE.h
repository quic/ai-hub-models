//
// Created by shivmahe on 9/5/2023.
//

#ifndef IMAGEENHANCEMENT_STABLELLVE_H
#define IMAGEENHANCEMENT_STABLELLVE_H

#include "Model.h"

class StableLLVE : public Model {

public:
    StableLLVE(){
        model_name = "StableLLVE";
    }

    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();


};


#endif //IMAGEENHANCEMENT_STABLELLVE_H
