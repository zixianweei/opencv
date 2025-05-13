#ifndef OPENCV_DNN_METAL_EXEC_OP_SOFTMAX_H
#define OPENCV_DNN_METAL_EXEC_OP_SOFTMAX_H

#include "../../precomp.hpp"
#include "op_base.h"

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

class OpSoftmax final : public OpBase
{
public:
    OpSoftmax() = default;
    explicit OpSoftmax(int axis, bool log_softmax) : axis_(axis), log_softmax_(log_softmax) {}

    bool forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) CV_OVERRIDE;

    void setAxis(int axis) { axis_ = axis; };
    int axis() const { return axis_; }

    void setLogSoftmax(bool log_softmax) { log_softmax_ = log_softmax; }
    bool logSoftmax() const { return log_softmax_; }

private:
    int axis_{-1};
    bool log_softmax_{false};
};

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_EXEC_OP_SOFTMAX_H
