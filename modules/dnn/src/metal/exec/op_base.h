#ifndef OPENCV_DNN_METAL_EXEC_OP_BASE_H
#define OPENCV_DNN_METAL_EXEC_OP_BASE_H

#include <vector>

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

class Tensor;

class OpBase
{
public:
    OpBase() = default;
    virtual ~OpBase() = default;

    OpBase(const OpBase&) = delete;
    OpBase& operator=(const OpBase&) = delete;
    OpBase(OpBase&&) noexcept = delete;
    OpBase& operator=(OpBase&&) noexcept = delete;

    virtual bool forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) = 0;
};

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_EXEC_OP_BASE_H
