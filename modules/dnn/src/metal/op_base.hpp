#ifndef OPENCV_DNN_METAL_OP_BASE_HPP
#define OPENCV_DNN_METAL_OP_BASE_HPP

#include <string>

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

class OpBase
{
public:
    OpBase(std::string name);

private:
    std::string name_;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_BASE_HPP
