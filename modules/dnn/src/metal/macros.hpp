#ifndef OPENCV_DNN_METAL_MACROS_HPP
#define OPENCV_DNN_METAL_MACROS_HPP

#if !__has_feature(objc_arc)
#define OCV_METAL_SAFE_RELEASE(__object__) \
    if (__object__ != nil) \
    { \
        [__object__ release]; \
    }
#else
#define OCV_METAL_SAFE_RELEASE(__object__) __object__ = nil;
#endif

#ifdef __OBJC__
#define OCV_METAL_FORWARD_DECLARATION(__class__) @class __class__
#else
#define OCV_METAL_FORWARD_DECLARATION(__class__) typedef struct objc_object __class__
#endif

#ifdef __OBJC__
#define OCV_METAL_TYPE_ALIAS(__true_type__, __alias_type__) \
    typedef __true_type__ __alias_type__
#else
#define OCV_METAL_TYPE_ALIAS(__true_type__, __alias_type__) \
    typedef void* __alias_type__
#endif

#endif  // !OPENCV_DNN_METAL_MACROS_HPP
