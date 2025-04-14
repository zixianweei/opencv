#include "tensor.hpp"
#include "context.hpp"

#ifdef HAVE_METAL
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#pragma mark (Private Utilities)

static int shapeCount(std::vector<int>& shape)
{
    int count = 1;
    for (int& e : shape)
    {
        count *= e;
    }
    return count;
}

static int elementSize(cv::dnn::metal::Format format)
{
    switch (format) {
        case cv::dnn::metal::Format::kFloat32:
            return 4;
            break;
        case cv::dnn::metal::Format::kUnsignedChar8:
            return 1;
        default:
            break;
    }
    // TODO(zixianwei): check here.
    return 0;
}

#pragma mark (TensorImpl)

@interface TensorImpl : NSObject
@property(assign, nonatomic) id<MTLBuffer> buffer;
@property(assign, nonatomic) std::vector<int> shape;
@property(assign, nonatomic) cv::dnn::metal::Format format;
- (instancetype)init;
- (BOOL)reshape:(const void*)data shape:(std::vector<int>&)shape format:(cv::dnn::metal::Format)format;
@end

@implementation TensorImpl

- (instancetype)init {
    self = [super init];
    if (self) {
        _buffer = nil;
        _shape = {};
        _format = cv::dnn::metal::Format::kUnknown;
    }
    return self;
}

- (BOOL)reshape:(const void*)data shape:(std::vector<int>&)shape format:(cv::dnn::metal::Format)format {
    id<MTLDevice> device = cv::dnn::metal::Context::getInstance().device();
    
    _shape = shape;
    _format = format;
    
    int length = shapeCount(_shape) * elementSize(_format);
    _buffer = [device newBufferWithLength:length options:MTLResourceStorageModeShared];
    
    memcpy(_buffer.contents, data, length);
    
    return TRUE;
}

- (BOOL)copyDataFromDevice:(void *)data {
    return TRUE;
}

@end

#endif // HAVE_METAL

#pragma mark (Tensor)

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

Tensor::Tensor()
{
    impl = [TensorImpl new];
}

Tensor::~Tensor()
{
    [impl release];
}

bool Tensor::reshape(const void* data, std::vector<int>& shape, Format format)
{
    return [impl reshape:data shape:shape format:format] == TRUE;
}

bool Tensor::copyDataFromDevice(void* data) const
{
    return [impl copyDataFromDevice:data] == TRUE;
}

#endif // HAVE_METAL
}}}
