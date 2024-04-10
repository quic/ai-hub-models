//
// Created by shivmahe on 9/5/2023.
//

#ifndef IMAGEENHANCEMENT_RUAS_H
#define IMAGEENHANCEMENT_RUAS_H

#include "Model.h"

class RUAS : public Model {

public:
    RUAS(){
        model_name = "RUAS";
    }

    void preprocess(cv::Mat &img, std::vector<int> dims);
    void postprocess(cv::Mat &outputimg);
    void msg();



};


#endif //IMAGEENHANCEMENT_RUAS_H
