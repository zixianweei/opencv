#include "tensor.hpp"

#ifdef HAVE_METAL
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

@interface TensorImpl : NSObject
@property(assign, nonatomic) id<MTLBuffer> buffer;
@property(assign, nonatomic) id<MTLTexture> texture;
@property(assign, nonatomic) NSUInteger size;
@property(assign, nonatomic) MTLDataType dataType;
@property(assign, nonatomic) MTLSize dimensions;
@end

@implementation TensorImpl

- (instancetype)init {
    self = [super init];
    if (self) {
        _size = 0;
        _dataType = MTLDataTypeFloat;
        _dimensions = MTLSizeMake(0, 0, 0);
    }
    return self;
}

- (BOOL)copyDataToDevice:(const void*)data {
    if (!self.buffer || !data)
        return FALSE;
    
    void* bufferContents = [self.buffer contents];
    if (!bufferContents) return FALSE;
    
    memcpy(bufferContents, data, self.size);
    return TRUE;
}

- (BOOL)copyDataFromDevice:(void *)data {
    if (!self.buffer || !data) return FALSE;
    
    void* bufferContents = [self.buffer contents];
    if (!bufferContents) return FALSE;
    
    memcpy(data, bufferContents, self.size);
    return TRUE;
}

- (void)allocateBufferWithDevice:(id<MTLDevice>)device size:(NSUInteger)size {
    self.buffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    self.size = size;
}

- (void)allocateTextureWithDevice:(id<MTLDevice>)device 
                        width:(NSUInteger)width 
                       height:(NSUInteger)height 
                       depth:(NSUInteger)depth {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    self.texture = [device newTextureWithDescriptor:desc];
    self.dimensions = MTLSizeMake(width, height, depth);
}

@end

#endif // HAVE_METAL

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

bool Tensor::copyDataToDevice(const void* data)
{
    return [impl copyDataToDevice:data] == TRUE;
}

bool Tensor::copyDataFromDevice(void* data) const
{
    return [impl copyDataFromDevice:data] == TRUE;
}

#endif // HAVE_METAL
}}}
