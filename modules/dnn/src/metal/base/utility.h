#ifndef OPENCV_DNN_METAL_BASE_UTILITY_H
#define OPENCV_DNN_METAL_BASE_UTILITY_H

#include "opencv2/dnn/dnn.hpp"
#include "tensor.h"

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

int shapeContent(const MatShape& shape, int axis);

int shapeCount(const MatShape &shape, int start_axis = 0, int end_axis = -1);

int elementSize(Format format);

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_BASE_UTILITY_H
