#include "context.hpp"

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

#include "opencv2/core/utils/logger.hpp"

static const char* const image_name = "libopencv_dnn.dylib";
static const char* const section_name = "opencv_metallib";

static dispatch_data_t find_section_data(const std::string& section_name)
{
    uint32_t image_idx = 0U;
    uint32_t image_count = _dyld_image_count();
    for (uint32_t i = 0U; i < image_count; i++) {
        if (strstr(_dyld_get_image_name(i), image_name)) {
            image_idx = i;
            break;
        }
    }

    const struct mach_header_64* image_header = reinterpret_cast<const struct mach_header_64*>(_dyld_get_image_header(image_idx));

    unsigned long section_size = 0;
    const uint8_t* section_data = getsectiondata(image_header, "__TEXT", section_name.c_str(), &section_size);
    if (section_data == nullptr) {
        throw std::runtime_error("Can't find metal library section " + section_name);
    }
    return dispatch_data_create(section_data,
        section_size,
        dispatch_get_main_queue(),
        ^() {});
}

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef NSMutableDictionary<NSString*, id<MTLComputePipelineState>>*
    ComputePipelineStateDictionary;
typedef NSMutableArray<id<MTLCommandBuffer>>* CommandBufferArray;

@interface ContextImpl : NSObject
@property(strong, nonatomic) id<MTLDevice> device;
@property(strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property(strong, nonatomic) id<MTLLibrary> library;
@property (strong, nonatomic) dispatch_queue_t queue;
@property(strong, nonatomic) ComputePipelineStateDictionary cachedCPS;

- (BOOL)isAvailable;
@end

@implementation ContextImpl

- (instancetype)init {
    self = [super init];
    
    if (_device == nil) {
        _device = MTLCreateSystemDefaultDevice();
        CV_LOG_INFO(NULL, "Device name: " << [[_device name] UTF8String] << ".");
    }
    if (_device == nil) {
        CV_LOG_ERROR(NULL, __func__ << ": error: failed to create metal device.");
        return nil;
    }
    
    if (_commandQueue == nil) {
        _commandQueue = [_device newCommandQueue];
    }
    if (_commandQueue == nil) {
        CV_LOG_ERROR(NULL, __func__ << ": error: failed to create command queue.");
        return nil;
    }

    if (_library == nil) {
        NSError* error;
        _library = [_device newLibraryWithData:find_section_data(section_name) error:&error];
        if (error) {
            CV_LOG_ERROR(NULL, __func__ << ": error: " << [[error description] UTF8String]);
            return nil;
        }
        CV_LOG_INFO(NULL, __func__ << ": create library from section");
    }
    
    return self;
}

- (BOOL)isAvailable {
    return _device != nil;
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

Context::~Context()
{
    OCV_METAL_SAFE_RELEASE(impl_);
}

bool Context::isAvailable()
{
    return [Context::getInstance().impl_ isAvailable] == TRUE;
}

Context::Context()
{
    impl_ = [[ContextImpl alloc] init];
}

bool isAvailable()
{
    return Context::getInstance().isAvailable();
}

MTLDevicePtr Context::device() {
    return [impl_ device];
}

#endif // HAVE_METAL
}}}
