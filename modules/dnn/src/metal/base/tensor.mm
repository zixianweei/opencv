#include "tensor.h"

#ifdef HAVE_METAL

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include "opencv2/core/check.hpp"
#include "opencv2/core/utils/logger.hpp"

#include "context.h"
#include "buffer.h"
#include "utility.h"

namespace cv { namespace dnn { namespace metal {

Tensor::Tensor(Format format) : buffer_(nullptr), format_(format), size_in_bytes_(0U)
{
}

Tensor::Tensor(const void *data, const MatShape& shape, Format format) : format_(format), size_in_bytes_(0U)
{
    this->upload(data, shape, format);
}

bool Tensor::upload(const void *data, const MatShape& shape, Format format)
{
    int size_in_bytes = shapeCount(shape) * elementSize(format);
    if (size_in_bytes <= 0)
    {
        CV_LOG_ERROR(NULL, __func__ << ": invalid buffer length.");
        return false;
    }

    if (size_in_bytes > size_in_bytes_)
    {
        buffer_.reset(new Buffer(data, size_in_bytes));
        if (buffer_ == nullptr)
        {
            CV_LOG_ERROR(NULL, __func__ << ": failed to create metal buffer, buffer size: " << size_in_bytes << ".");
            return false;
        }
    }

    shape_ = shape;
    format_ = format;
    size_in_bytes_ = size_in_bytes;

    if (data != nullptr)
    {
        std::ignore = memcpy(buffer_->rawBuffer().contents, data, size_in_bytes);
    }

    return true;
}

bool Tensor::download(void *data, const MatShape& shape, Format format)
{
    // tensor content is downloading to null pointer.
    CV_Assert(data != nullptr);
    int size_in_bytes = shapeCount(shape) * elementSize(format);
    if (size_in_bytes > size_in_bytes_)
    {
        CV_LOG_ERROR(NULL, __func__ << ": data is larger than buffer, buffer size: " << size_in_bytes << ".");
        return false;
    }

    std::ignore = memcpy(data, buffer_->rawBuffer().contents, size_in_bytes);
    return true;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
