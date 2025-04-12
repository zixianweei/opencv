#include "context.hpp"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>

#include <mutex>

@interface ContextImpl : NSObject
@property(assign, nonatomic) id<MTLDevice> device;
@property(assign, nonatomic) id<MTLCommandQueue> commandQueue;
@end

@interface ContextImpl ()
@property(strong, nonatomic) id<MTLLibrary> library;
//@property(strong, nonatomic) NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cachedPSO;
//@property(strong, nonatomic) NSMutableArray<id<MTLCommandBuffer>>* waitingCommandBuffer;
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

@end

#endif // HAVE_METAL

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

static std::shared_ptr<Context> g_ctx;
static std::once_flag g_flag;

// static
std::shared_ptr<Context> Context::create()
{
    std::call_once(g_flag, []() {
        g_ctx = std::shared_ptr<Context>(new Context());
    });
    return g_ctx;
}

Context::Context()
{
    context_impl_ = [ContextImpl new];
}

#endif // HAVE_METAL
}}}
