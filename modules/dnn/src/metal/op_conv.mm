#include "op_conv.hpp"

#ifdef HAVE_METAL
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

@interface OpConvImpl : NSObject

@end

@implementation OpConvImpl

- (instancetype)init {
    self = [super init];
    return self;
    
    // create tensor input output
    // copy data from input tensor, host to device
    // forward conv -> shader msl metal shader language
    // copy data to output tensor, device to host
    // set dirty
    // retrive output and reshape
}

@end

#endif // HAVE_METAL

namespace cv { namespace dnn {namespace metal {
#ifdef HAVE_METAL

OpConv::OpConv() {
    impl_ = [OpConvImpl new];
}

bool OpConv::forward(std::vector<Tensor>& ins, std::vector<Tensor>& outs)
{
    return false;
}

#endif // HAVE_METAL
}}}
