#ifndef OPENCV_DNN_METAL_CONTEXT_HPP
#define OPENCV_DNN_METAL_CONTEXT_HPP

#include <memory>

#ifdef HAVE_METAL
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
@class ContextImpl;
#else
typedef struct objc_object ContextImpl;
#endif
#endif

namespace cv { namespace dnn { namespace metal {

#ifdef HAVE_METAL

class Context {
public:
    static std::shared_ptr<Context> create();

    Context(const Context&) = delete;
    Context& operator=(const Context) = delete;
    Context(Context&&) noexcept = delete;
    Context& operator=(Context&&) noexcept = delete;
    
    bool isAvailable();

private:
    Context();

    __strong ContextImpl* context_impl_{};
};

#endif // HAVE_METAL

}}}

#endif // !OPENCV_DNN_METAL_CONTEXT_HPP
