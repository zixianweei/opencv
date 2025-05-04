#ifndef OPENCV_DNN_METAL_OP_SOFTMAX_HPP
#define OPENCV_DNN_METAL_OP_SOFTMAX_HPP

#include "../precomp.hpp"
#include "op_base.hpp"
#include "macros.hpp"

OCV_METAL_FORWARD_DECLARATION(OpSoftmaxImpl);

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

struct OpSoftmaxAttribute
{
    int axis = 0;
    int axis_bias = 0;
    int axis_step = 0;
    bool log_softmax = false;
};

class OpSoftmax final : public OpBase {
public:
    OpSoftmax();
    ~OpSoftmax();

    bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;

    void setAxis(int axis);
    void setAxisBias(int axis_bias);
    void setAxisStep(int axis_step);
    void setLogSoftmax(bool log_softmax);

private:
    OpSoftmaxImpl* impl_;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_SOFTMAX_HPP
