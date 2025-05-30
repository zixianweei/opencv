#include "buffer.h"

#include "opencv2/core/check.hpp"

#include "context.h"

#ifdef HAVE_METAL

@implementation MTL4DNNBuffer

- (instancetype)initWithData:(const void *)data size:(size_t)size
{
    self = [super init];
    if (self)
    {
        _buffer = [[MPS4DNN_CONTEXT device] newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    }
    return self;
}

- (void)dealloc
{
    OCV_DNN_METAL_SAFE_RELEASE(_buffer);
    [super dealloc];
}

@end

namespace cv { namespace dnn { namespace metal {

Buffer::Buffer(const void *data, size_t size)
{
    buffer_ = [[MTL4DNNBuffer alloc] initWithData:data size:size];
}

Buffer::~Buffer()
{
    OCV_DNN_METAL_SAFE_RELEASE(buffer_);
}

id<MTLBuffer> Buffer::rawBuffer()
{
    CV_CheckTrue(buffer_ != nil, "metal buffer is nil.");
    return [buffer_ buffer];
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
