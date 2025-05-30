#ifndef OPENCV_DNN_METAL_BASE_UTILITY_H
#define OPENCV_DNN_METAL_BASE_UTILITY_H

#ifdef HAVE_METAL

#include "opencv2/dnn/dnn.hpp"
#include "tensor.h"

#include <Foundation/Foundation.h>

OCV_DNN_METAL_OBJC_PRIVATE_GUARD();

namespace cv { namespace dnn { namespace metal {

typedef NSArray<NSNumber*> MPSShape;

int shapeContent(const MatShape& shape, int axis);

int shapeCount(const MatShape &shape, int start_axis = 0, int end_axis = -1);

int elementSize(Format format);

MPSShape* makeMPSShape(const std::vector<int>& shape);

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_BASE_UTILITY_H
