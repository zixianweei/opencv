#ifndef OPENCV_DNN_METAL_EXEC_OP_CONV_H
#define OPENCV_DNN_METAL_EXEC_OP_CONV_H

#include "../../precomp.hpp"
#include "op_base.h"

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

class OpConv final : public OpBase
{
public:
    OpConv(cv::Mat weights, int nGroups, int strideInX, int strideInY, int paddingLeft, int paddingRight, int paddingTop, int paddingBottom, int dilationRateInX, int dilationRateInY)
        : weights_(weights), nGroups_(nGroups), strideInX_(strideInX), strideInY_(strideInY),
        paddingLeft_(paddingLeft), paddingRight_(paddingRight), paddingTop_(paddingTop), paddingBottom_(paddingBottom),
        dilationRateInX_(dilationRateInX), dilationRateInY_(dilationRateInY) {}

    bool forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) CV_OVERRIDE;

private:
    cv::Mat weights_;
    uint nGroups_;
    uint strideInX_;
    uint strideInY_;
    uint paddingLeft_;
    uint paddingRight_;
    uint paddingTop_;
    uint paddingBottom_;
    uint dilationRateInX_;
    uint dilationRateInY_;
};

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_EXEC_OP_CONV_H
