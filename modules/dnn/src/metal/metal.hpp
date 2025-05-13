#ifndef OPENCV_DNN_METAL_HPP
#define OPENCV_DNN_METAL_HPP

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

bool isAvailable();

}}} // namespace cv::dnn::metal

#include "base/tensor.h"

#include "exec/op_base.h"
#include "exec/op_softmax.h"

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_HPP
