#include "op_softmax.hpp"
#include "tensor.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@interface OpSoftmaxImpl : NSObject
@property (assign, nonatomic) int axis;
@property (assign, nonatomic) int axisBias;
@property (assign, nonatomic) int axisStep;
@property (assign, nonatomic) bool logSoftmax;

- (instancetype)init;
@end

@implementation OpSoftmaxImpl

- (instancetype)init
{
    self = [super init];
    return self;
}

@end

namespace cv { namespace dnn {namespace metal {
#ifdef HAVE_METAL

OpSoftmax::OpSoftmax()
{
    impl_ = [[OpSoftmaxImpl alloc] init];
}

OpSoftmax::~OpSoftmax()
{
    OCV_METAL_SAFE_RELEASE(impl_);
}

bool OpSoftmax::forward(std::vector<Tensor> &ins, std::vector<Tensor> &outs)
{
    CV_Assert(ins.size() == 1U && outs.size() == 1U);
    return false;
}

void OpSoftmax::setAxis(int axis)
{
    [impl_ setAxis:axis];
}

void OpSoftmax::setAxisBias(int axis_bias)
{
    [impl_ setAxisBias:axis_bias];
}

void OpSoftmax::setAxisStep(int axis_step)
{
    [impl_ setAxisStep:axis_step];
}

void OpSoftmax::setLogSoftmax(bool log_softmax)
{
    [impl_ setLogSoftmax:log_softmax];
}

#endif // HAVE_METAL
}}}
