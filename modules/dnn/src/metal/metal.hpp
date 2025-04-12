#ifndef OPENCV_DNN_METAL_HPP
#define OPENCV_DNN_METAL_HPP

#include "op_base.hpp"

namespace cv { namespace dnn { namespace metal {

#ifdef HAVE_METAL

bool isAvailable();

#endif // HAVE_METAL

}}}

#include "context.hpp"
#include "tensor.hpp"

#endif // !OPENCV_DNN_METAL_HPP
