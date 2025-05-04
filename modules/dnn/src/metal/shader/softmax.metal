#include <metal_stdlib>
#include "../op_softmax.hpp"

using namespace metal;

kernel void kernel_softmax(const device float* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant cv::dnn::metal::OpSoftmaxProperty &property [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]])
{
    
}


kernel void kernel_softmax_axis_2(const device float* src [[buffer(0)]],
                                  device float* dst [[buffer(1)]],
                                  constant cv::dnn::metal::OpSoftmaxProperty &property [[buffer(2)]],
                                  uint3 gid [[thread_position_in_grid]])
{
    
}
