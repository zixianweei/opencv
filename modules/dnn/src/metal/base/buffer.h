#ifndef OPENCV_DNN_METAL_BASE_BUFFER_H
#define OPENCV_DNN_METAL_BASE_BUFFER_H

#include "macros.h"

#ifdef HAVE_METAL

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

OCV_DNN_METAL_OBJC_PRIVATE_GUARD();

@interface MTL4DNNBuffer : NSObject
@property(strong, nonatomic) id<MTLBuffer> buffer;

- (instancetype)initWithData:(const void *)data size:(size_t)size;
- (void)dealloc;
@end

namespace cv { namespace dnn { namespace metal {

class Buffer
{
public:
    Buffer(const void *data, size_t size);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept = delete;
    Buffer& operator=(Buffer&&) noexcept = delete;

    id<MTLBuffer> rawBuffer();

private:
    MTL4DNNBuffer* buffer_{nullptr};
};

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL

#endif // !OPENCV_DNN_METAL_BASE_BUFFER_H
