#include "tensor.hpp"
#include "context.hpp"

#ifdef HAVE_METAL

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include "opencv2/core/utils/logger.hpp"

namespace
{

int shapeCount(cv::dnn::MatShape& shape)
{
    int count = 1;
    for (int e : shape) {
        count *= e;
    }
    return count;
}

int elementSize(cv::dnn::metal::Format format)
{
    switch (format) {
    case cv::dnn::metal::Format::kFloat32:
        return 4;
        break;
    case cv::dnn::metal::Format::kFloat16:
        return 2;
        break;
    case cv::dnn::metal::Format::kUnsignedChar8:
        return 1;
    case cv::dnn::metal::Format::kUnknown:
    default:
        CV_LOG_FATAL(NULL, __func__ << ": invalid element size. format: " << static_cast<int>(format));
        break;
    }
    return 0;
}

}

@interface TensorImpl : NSObject
@property (strong, nonatomic) id<MTLBuffer> buffer;
@property (assign, nonatomic) cv::dnn::MatShape shape;
@property (assign, nonatomic) cv::dnn::metal::Format format;
@property (assign, nonatomic) int sizeInBytes;

- (instancetype)init;
- (instancetype)initWithFormat:(cv::dnn::metal::Format)format;
- (void)dealloc;
- (BOOL)fromBytes:(const void*)data
            shape:(cv::dnn::MatShape)shape
           format:(cv::dnn::metal::Format) format;
- (BOOL)toBytes:(void**)data
          shape:(cv::dnn::MatShape)shape
         format:(cv::dnn::metal::Format)format;
@end

@implementation TensorImpl

- (instancetype)init
{
    self = [super init];
    if (self) {
        _buffer = nil;
        _shape = {};
        _format = cv::dnn::metal::Format::kUnknown;
    }
    return self;
}

- (instancetype)initWithFormat:(cv::dnn::metal::Format)format
{
    self = [super init];
    if (self) {
        _buffer = nil;
        _shape = {};
        _format = format;
    }
    return self;
}

- (void)dealloc
{
    OCV_METAL_SAFE_RELEASE(_buffer);
    [super dealloc];
}

- (BOOL)fromBytes:(const void*)data
            shape:(cv::dnn::MatShape)shape
           format:(cv::dnn::metal::Format) format
{
    id<MTLDevice> device = cv::dnn::metal::Context::getInstance().device();
    if (device == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": metal context device is nil.");
        return FALSE;
    }

    [self setShape:shape];
    [self setFormat:format];

    int dataLen = shapeCount(_shape) * elementSize(_format);
    if (dataLen <= 0)
    {
        CV_LOG_ERROR(NULL, __func__ << ": metal invalid buffer length. length = " << dataLen);
        return FALSE;
    }

    // TODO: @zixianweei Hint, should release origin buffer or not. Please check this.
    if (dataLen > [self sizeInBytes])
    {
        OCV_METAL_SAFE_RELEASE([self buffer]);
        [self setBuffer:[device newBufferWithLength:dataLen options:MTLResourceStorageModeShared]];
        if ([self buffer] == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": metal buffer is nil.");
            return FALSE;
        }
    }

    [self setSizeInBytes:dataLen];

    if (data != nullptr)
    {
        std::ignore = memcpy([self buffer].contents, data, dataLen);
    }

    return TRUE;
}

- (BOOL)toBytes:(void**)data
          shape:(cv::dnn::MatShape)shape
         format:(cv::dnn::metal::Format)format
{
    int dataLen = shapeCount(shape) * elementSize(format);
    if (dataLen > [self sizeInBytes])
    {
        CV_LOG_ERROR(NULL, __func__ << ": data is larger than buffer. data length = " << dataLen << ".");
        return FALSE;
    }

    std::ignore = memcpy(*data, [self buffer].contents, dataLen);
    return TRUE;
}

@end

#endif // HAVE_METAL

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

Tensor::Tensor(Format format)
{
    impl_ = [[TensorImpl alloc] initWithFormat:format];
}

Tensor::Tensor(const void* data, MatShape& shape, Format format)
{
    impl_ = [[TensorImpl alloc] init];
    [impl_ fromBytes:data shape:shape format:format];
}

Tensor::~Tensor()
{
    OCV_METAL_SAFE_RELEASE(impl_);
}

bool Tensor::fromBytes(const void *data, MatShape &shape, Format format)
{
    return [impl_ fromBytes:data shape:shape format:format];
}

bool Tensor::toBytes(void** data, MatShape& shape, Format format)
{
    return [impl_ toBytes:data shape:shape format:format];
}

MatShape Tensor::getShape() const
{
    return [impl_ shape];
}

MTLBufferPtr Tensor::getRawBuffer()
{
    return [impl_ buffer];
}

#endif // HAVE_METAL
}}}
