#include "context.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>

@interface ContextImpl : NSObject
@property(assign, nonatomic) id<MTLDevice> device;
@property(assign, nonatomic) id<MTLCommandQueue> commandQueue;
@property(strong, nonatomic) id<MTLLibrary> library;
- (BOOL)isAvailable;
- (id<MTLDevice>)device;
@end

@implementation ContextImpl

+ (id<MTLDevice>) sharedDevice {
    static id<MTLDevice> s_shared_device = nil;
    static dispatch_once_t token;
    dispatch_once(&token, ^{
        s_shared_device = MTLCreateSystemDefaultDevice();
    });
    return s_shared_device;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _device = [ContextImpl sharedDevice];
        _commandQueue = [_device newCommandQueue];
    }
    return self;
}

- (BOOL)isAvailable {
    return _device != nil;
}

- (id<MTLDevice>)device {
    return _device;
}

@end

#endif // HAVE_METAL

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

// static
Context& Context::getInstance()
{
    static Context instance;
    return instance;
}

bool Context::isAvailable()
{
    return [Context::getInstance().context_impl_ isAvailable] == TRUE;
}

Context::Context()
{
    context_impl_ = [ContextImpl new];
}

bool isAvailable()
{
    return Context::getInstance().isAvailable();
}

MTLDeviceType Context::device() {
    return [context_impl_ device];
}

#endif // HAVE_METAL
}}}
