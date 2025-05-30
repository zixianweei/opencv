#include "context.h"

#ifdef HAVE_METAL

#include <string>

#include <mach-o/dyld.h>
#include <mach-o/getsect.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "opencv2/core/utils/logger.hpp"

@implementation MPS4DNNContext

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

    return self;
}

- (void)dealloc
{
    OCV_DNN_METAL_SAFE_RELEASE(_device);
    OCV_DNN_METAL_SAFE_RELEASE(_commandQueue);
    [super dealloc];
}

@end

namespace cv { namespace dnn { namespace metal {

// static
MPS4DNNContextOwner& MPS4DNNContextOwner::getInstance()
{
    static MPS4DNNContextOwner instance;
    return instance;
}

MPS4DNNContextOwner::~MPS4DNNContextOwner()
{
    OCV_DNN_METAL_SAFE_RELEASE(context_);
}

MPS4DNNContextOwner::MPS4DNNContextOwner()
{
    context_ = [[MPS4DNNContext alloc] init];
    if (context_ == nullptr)
    {
        CV_LOG_ERROR(NULL, __func__ << ": failed to create mps context");
    }
}

// declaration in |metal.hpp|
bool isAvailable()
{
    return [MPS4DNN_CONTEXT device] != nil;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
