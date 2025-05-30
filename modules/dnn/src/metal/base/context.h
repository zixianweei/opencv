#ifndef OPENCV_DNN_METAL_BASE_CONTEXT_H
#define OPENCV_DNN_METAL_BASE_CONTEXT_H

#include <string>

#include "macros.h"

#ifdef HAVE_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

OCV_DNN_METAL_OBJC_PRIVATE_GUARD();

@interface MPS4DNNContext : NSObject
@property(strong, nonatomic) id<MTLDevice> device;
@property(strong, nonatomic) id<MTLCommandQueue> commandQueue;

- (instancetype)init;
- (void)dealloc;
@end

namespace cv { namespace dnn { namespace metal {

class MPS4DNNContextOwner
{
public:
    static MPS4DNNContextOwner& getInstance();

    ~MPS4DNNContextOwner();
    MPS4DNNContextOwner(const MPS4DNNContextOwner&) = delete;
    MPS4DNNContextOwner& operator=(const MPS4DNNContextOwner&) = delete;
    MPS4DNNContextOwner(MPS4DNNContextOwner&&) noexcept = delete;
    MPS4DNNContextOwner& operator=(MPS4DNNContextOwner&&) noexcept = delete;

    MPS4DNNContext *context() { return context_; }

private:
    MPS4DNNContextOwner();

    MPS4DNNContext* context_{nullptr};
};

}}} // namespace cv::dnn::metal

#define MPS4DNN_CONTEXT cv::dnn::metal::MPS4DNNContextOwner::getInstance().context()

#endif  // HAVE_METAL

#endif  // !OPENCV_DNN_METAL_BASE_CONTEXT_H
