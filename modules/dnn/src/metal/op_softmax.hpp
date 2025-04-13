#ifndef OPENCV_DNN_METAL_OP_SOFTMAX_HPP
#define OPENCV_DNN_METAL_OP_SOFTMAX_HPP

#include "../precomp.hpp"
#include "op_base.hpp"

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

class OpSoftmax final : public OpBase {
public:
    OpSoftmax() = default;
    
    bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_SOFTMAX_HPP
