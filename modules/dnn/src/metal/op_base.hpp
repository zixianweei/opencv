#ifndef OPENCV_DNN_METAL_OP_BASE_HPP
#define OPENCV_DNN_METAL_OP_BASE_HPP

#include <string>
#include <vector>

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

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

    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) = 0;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_BASE_HPP
