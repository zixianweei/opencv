#ifndef OPENCV_DNN_METAL_CONTEXT_HPP
#define OPENCV_DNN_METAL_CONTEXT_HPP

#ifdef HAVE_METAL
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
@class ContextImpl;
typedef id<MTLDevice> MTLDeviceType;
typedef id<MTLCommandBuffer> MTLCommandBufferType;
#else
typedef struct objc_object ContextImpl;
typedef void* MTLDeviceType;
typedef void* MTLCommandBufferType;
#endif
#endif

namespace cv { namespace dnn { namespace metal {

#ifdef HAVE_METAL

class Context {
public:
    static Context& getInstance();

    Context(const Context&) = delete;
    Context& operator=(const Context) = delete;
    Context(Context&&) noexcept = delete;
    Context& operator=(Context&&) noexcept = delete;
    
    bool isAvailable();
    MTLDeviceType device();

private:
    Context();

    __strong ContextImpl* context_impl_{};
};

#endif // HAVE_METAL

}}}

#endif // !OPENCV_DNN_METAL_CONTEXT_HPP
