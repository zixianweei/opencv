#ifndef OPENCV_DNN_METAL_TENSOR_HPP
#define OPENCV_DNN_METAL_TENSOR_HPP

#include <cstdint>
#include <vector>

#ifdef HAVE_METAL
#ifdef __OBJC__
@class TensorImpl;
#else
typedef struct objc_object TensorImpl;
#endif
#endif

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

enum class Format
{
    kUnknown,
    kUnsignedChar8,
    kFloat32,
};

class Tensor
{
public:
    Tensor();
    ~Tensor();
    
    bool reshape(const void* data, std::vector<int>& shape, Format format = Format::kFloat32);
    bool copyDataFromDevice(void* data) const;

private:
    __strong TensorImpl* impl;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_TENSOR_HPP
