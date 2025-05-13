#ifndef OPENCV_DNN_METAL_BASE_CONTEXT_H
#define OPENCV_DNN_METAL_BASE_CONTEXT_H

#include <string>

#include "macros.h"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

OCV_DNN_METAL_OBJC_PRIVATE_GUARD();

typedef NSMutableDictionary<NSString *, id<MTLComputePipelineState>>* ComputePipelineStateDictionary;

@interface MTL4DNNContext : NSObject
@property(strong, nonatomic) id<MTLDevice> device;
@property(strong, nonatomic) id<MTLCommandQueue> commandQueue;
@property(strong, nonatomic) id<MTLLibrary> library;
@property(strong, nonatomic) ComputePipelineStateDictionary cachedCPS;
@property(strong, nonatomic) id<MTLCommandBuffer> commandBuffer;

- (instancetype)init;
- (void)dealloc;
- (id<MTLComputePipelineState>)findComputePipelineState:(NSString *)kernelName;
- (id<MTLComputeCommandEncoder>)makeEncoder;
- (BOOL)commit;
@end

namespace cv { namespace dnn { namespace metal {

class ContextOwner
{
public:
    static ContextOwner& getInstance();

    ~ContextOwner();
    ContextOwner(const ContextOwner&) = delete;
    ContextOwner& operator=(const ContextOwner&) = delete;
    ContextOwner(ContextOwner&&) noexcept = delete;
    ContextOwner& operator=(ContextOwner&&) noexcept = delete;

    MTL4DNNContext *context() { return context_; }

private:
    ContextOwner();

    MTL4DNNContext* context_{nullptr};
};

}}} // namespace cv::dnn::metal

#define MTL4DNN_CONTEXT cv::dnn::metal::ContextOwner::getInstance().context()

#endif  // HAVE_METAL

#endif  // !OPENCV_DNN_METAL_BASE_CONTEXT_H
