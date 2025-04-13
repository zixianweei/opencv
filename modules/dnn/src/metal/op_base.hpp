#ifndef OPENCV_DNN_METAL_OP_BASE_HPP
#define OPENCV_DNN_METAL_OP_BASE_HPP

#include <string>

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

class Tensor;

class OpBase
{
public:
    OpBase() = default;
    virtual ~OpBase() = default;
    virtual bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) = 0;
    
private:
    std::string shader_name_;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_BASE_HPP
