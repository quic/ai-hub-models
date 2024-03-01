//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorShape.hpp"

template <typename Container> Container& split(Container& result, const typename Container::value_type & s, typename Container::value_type::value_type delimiter )
{
  result.clear();
  std::istringstream ss( s );
  while (!ss.eof())
  {
    typename Container::value_type field;
    getline( ss, field, delimiter );
    if (field.empty()) continue;
    result.push_back( field );
  }
  return result;
}


cv::Mat get_affine_transform(int dst_w, int dst_h, int inv, double center[], double scale[]);
//void getcenterscale(int image_width, int image_height, double center[2], double scale[2]);
void getcenterscale(int image_width, int image_height, double center[2], double scale[2],float bottom, float left, float top, float right);
float** getCoords(std::vector<float32_t> buff, double center[], double scale[]);

#endif

