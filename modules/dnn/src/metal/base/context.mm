#include "context.h"

#ifdef HAVE_METAL

#include <string>

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "opencv2/core/utils/logger.hpp"

#if defined(OCV_DNN_USE_EMBED_METALLIB)
static const char* const image_name = "libopencv_dnn.4.12.0.dylib";
static const char* const section_name = "opencv_metallib";
#else
static const char* const standalone_metallib_name = "/Users/wzx/source/opencv/build/OpenCV.metallib";
#endif

#if defined(OCV_DNN_USE_EMBED_METALLIB)
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
#endif

@implementation MTL4DNNContext

- (instancetype)init
{
    self = [super init];

    CV_LOG_INFO(NULL, __func__ << ": allocating metal context.");
#if !defined(OCV_DNN_METAL_DEBUG)
    NSArray* devices = MTLCopyAllDevices();
    for (id<MTLDevice> deivce in devices)
    {
        CV_LOG_INFO(NULL, __func__ << ": found device: " << [[deivce name] UTF8String] << ".");
    }
#if !__has_feature(objc_arc)
    [devices autorelease];
#endif
#endif // OCV_DNN_METAL_DEBUG

    if (_device == nil)
    {
        _device = MTLCreateSystemDefaultDevice();
        CV_LOG_INFO(NULL, __func__ << ": using device: " << [[_device name] UTF8String] << ".");
    }

    if (_device == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": error: failed to find metal device.");
        return nil;
    }

    if (_commandQueue == nil)
    {
        _commandQueue = [_device newCommandQueue];
    }

    if (_commandQueue == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": error: failed to create command queue.");
        return nil;
    }

    if (_library == nil)
    {
        NSError* error;
#if defined(OCV_DNN_USE_EMBED_METALLIB)
        _library = [_device newLibraryWithData:find_section_data(section_name) error:&error];
#else
        _library = [_device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithFormat:@"%s", standalone_metallib_name]] error:&error];
#endif
        if (error)
        {
            CV_LOG_ERROR(NULL, __func__ << ": error: " << [[error description] UTF8String]);
            return nil;
        }
        CV_LOG_INFO(NULL, __func__ << ": initialized metal library successfully.");
    }

    if (_commandBuffer == nil)
    {
        _commandBuffer = [_commandQueue commandBuffer];
    }

    if (_commandBuffer == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": error: failed to create command buffer.");
        return nil;
    }

    return self;
}

- (void)dealloc
{
    OCV_DNN_METAL_SAFE_RELEASE(_device);
    OCV_DNN_METAL_SAFE_RELEASE(_commandQueue);
    OCV_DNN_METAL_SAFE_RELEASE(_library);
    OCV_DNN_METAL_SAFE_RELEASE(_cachedCPS);
    OCV_DNN_METAL_SAFE_RELEASE(_commandBuffer);
    [super dealloc];
}

- (id<MTLComputePipelineState>)findComputePipelineState:(NSString *)kernelName
{
    if (kernelName == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": error: kernel name is nil.");
        return nil;
    }

    id<MTLComputePipelineState> computePipelineState = _cachedCPS[kernelName];
    if (computePipelineState != nil)
    {
        CV_LOG_INFO(NULL, __func__ << ": found cached compute pipeline state: " << [kernelName UTF8String] << ".");
        return computePipelineState;
    }

    id<MTLFunction> function = [_library newFunctionWithName:kernelName];
    if (function == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": error: failed to create function: " << [kernelName UTF8String] << ".");
        return nil;
    }

    NSError* error;
    computePipelineState = [_device newComputePipelineStateWithFunction:function error:&error];
    if (error != nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": error: " << [[error description] UTF8String]);
        return nil;
    }

    _cachedCPS[kernelName] = computePipelineState;
    return computePipelineState;
}

- (id<MTLComputeCommandEncoder>)makeEncoder
{
    if (_commandBuffer == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": error: command buffer is nil.");
        return nil;
    }
    [_commandBuffer enqueue];
    return [_commandBuffer computeCommandEncoder];
}

- (BOOL)commit
{
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    _commandBuffer = [_commandQueue commandBuffer];
    return TRUE;
}

@end

namespace cv { namespace dnn { namespace metal {

// static
ContextOwner& ContextOwner::getInstance()
{
    static ContextOwner instance;
    return instance;
}

ContextOwner::~ContextOwner()
{
    OCV_DNN_METAL_SAFE_RELEASE(context_);
}

ContextOwner::ContextOwner()
{
    context_ = [[MTL4DNNContext alloc] init];
    if (context_ == nullptr)
    {
        CV_LOG_ERROR(NULL, __func__ << ": failed to create metal context");
    }
}

// declaration in |metal.hpp|
bool isAvailable()
{
    return [MTL4DNN_CONTEXT device] != nil;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
