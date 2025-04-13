#ifndef OPENCV_DNN_METAL_OP_CONV_HPP
#define OPENCV_DNN_METAL_OP_CONV_HPP

#include "../precomp.hpp"
#include "op_base.hpp"

#ifdef HAVE_METAL
#ifdef __OBJC__
@class OpConvImpl;
#else
typedef struct objc_object OpConvImpl;
#endif // __OBJ_C__
#endif // HAVE_METAL

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

class OpConv final : public OpBase
{
public:
    OpConv();
    
    bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;
    
private:
    __strong OpConvImpl* impl_;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_CONV_HPP
