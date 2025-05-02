#ifndef OPENCV_DNN_METAL_OP_CONV_HPP
#define OPENCV_DNN_METAL_OP_CONV_HPP

#include "../precomp.hpp"
#include "op_base.hpp"
#include "macros.hpp"

OCV_METAL_FORWARD_DECLARATION(OpConvImpl);

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

class OpConv final : public OpBase
{
public:
    OpConv();
    ~OpConv();
    
    bool forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs) CV_OVERRIDE;

    bool allocOpProperty(const OpProperty* property) CV_OVERRIDE;
    
private:
    OpConvImpl* impl_;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_OP_CONV_HPP
