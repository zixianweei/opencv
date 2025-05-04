#ifndef OPENCV_DNN_METAL_CONTEXT_HPP
#define OPENCV_DNN_METAL_CONTEXT_HPP

#include <string>

#include "macros.hpp"

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

OCV_METAL_FORWARD_DECLARATION(ContextImpl);

OCV_METAL_TYPE_ALIAS(id<MTLDevice>, MTLDevicePtr);
OCV_METAL_TYPE_ALIAS(id<MTLCommandBuffer>, MTLCommandBufferPtr);
OCV_METAL_TYPE_ALIAS(id<MTLComputePipelineState>, MTLComputePipelineStatePtr);
OCV_METAL_TYPE_ALIAS(id<MTLComputeCommandEncoder>, MTLComputeCommandEncoderPtr);

namespace cv { namespace dnn { namespace metal {

#ifdef HAVE_METAL

class Context
{
public:
    static Context& getInstance();

    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context) = delete;
    Context(Context&&) noexcept = delete;
    Context& operator=(Context&&) noexcept = delete;

    bool isAvailable();

    MTLDevicePtr device();
    MTLCommandBufferPtr commandQueue();
    MTLComputePipelineStatePtr findComputePipelineState(const std::string& kname);
    MTLComputeCommandEncoderPtr commandEncoder();
    bool commit();

private:
    Context();

    ContextImpl* impl_{nullptr};
};

#endif  // HAVE_METAL

}}}  // namespace cv::dnn::metal

#endif  // !OPENCV_DNN_METAL_CONTEXT_HPP
