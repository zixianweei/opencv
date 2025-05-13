#ifndef OPENCV_DNN_METAL_TENSOR_HPP
#define OPENCV_DNN_METAL_TENSOR_HPP

#include <cstdint>
#include <vector>
#include <memory>

#include "opencv2/dnn/dnn.hpp"

#include "macros.h"

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

enum class Format
{
    kUnknown,
    kUnsignedChar8,
    kFloat32,
    kFloat16,
};

class Buffer;

class Tensor
{
public:
    Tensor(Format format = Format::kFloat32);
    Tensor(const void* data, const MatShape& shape, Format format = Format::kFloat32);

    bool upload(const void* data, const MatShape& shape, Format format = Format::kFloat32);
    bool download(void* data, const MatShape& shape, Format format = Format::kFloat32);

    MatShape shape() const { return shape_; };
    size_t dims() const { return shape().size(); };

    std::shared_ptr<Buffer> buffer() const { return buffer_; }

private:
    std::shared_ptr<Buffer> buffer_;
    MatShape shape_;
    Format format_;
    size_t size_in_bytes_;
};

}}}  // namespace cv::dnn::metal

#endif  // HAVE_METAL

#endif  // !OPENCV_DNN_METAL_TENSOR_HPP
