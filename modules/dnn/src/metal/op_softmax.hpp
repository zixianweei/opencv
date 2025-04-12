#ifndef OPENCV_DNN_METAL_OP_SOFTMAX_HPP
#define OPENCV_DNN_METAL_OP_SOFTMAX_HPP

#include "op_base.hpp"

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

class OpSoftmax final : public OpBase {
public:
    OpSoftmax(std::string name) : OpBase(std::move(name)) {}
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_SOFTMAX_HPP
