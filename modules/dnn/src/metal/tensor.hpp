#ifndef OPENCV_DNN_METAL_TENSOR_HPP
#define OPENCV_DNN_METAL_TENSOR_HPP

#include <cstdint>
#include <vector>

#include "opencv2/dnn/dnn.hpp"

#include "macros.hpp"

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#endif

OCV_METAL_FORWARD_DECLARATION(TensorImpl);
OCV_METAL_TYPE_ALIAS(id<MTLBuffer>, MTLBufferPtr);

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

enum class Format
{
    kUnknown,
    kUnsignedChar8,
    kFloat32,
    kFloat16,
};

class Tensor
{
public:
    Tensor(Format format = Format::kFloat32);
    Tensor(const void* data, MatShape& shape, Format format = Format::kFloat32);
    ~Tensor();

    Tensor(const Tensor& rhs);
    Tensor& operator=(const Tensor& rhs);
    Tensor(Tensor&& rhs) noexcept;
    Tensor& operator=(Tensor&& rhs) noexcept;

    bool fromBytes(const void* data, MatShape& shape, Format format = Format::kFloat32);
    bool toBytes(void** data, MatShape& shape, Format format = Format::kFloat32);
    MatShape getShape() const;
    MTLBufferPtr getRawBuffer();

private:
    TensorImpl* impl_;
};

#endif  // HAVE_METAL
}}}  // namespace cv::dnn::metal

#endif  // !OPENCV_DNN_METAL_TENSOR_HPP
