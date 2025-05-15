#ifndef OPENCV_DNN_METAL_EXEC_OP_MATMUL_H
#define OPENCV_DNN_METAL_EXEC_OP_MATMUL_H

#include "../../precomp.hpp"
#include "op_base.h"

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

class OpMatMul final : public OpBase
{
public:
    OpMatMul() = default;

    bool forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) CV_OVERRIDE;

private:
};

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_EXEC_OP_MATMUL_H
