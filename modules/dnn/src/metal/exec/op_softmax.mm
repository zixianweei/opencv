#include "op_softmax.h"

#include "opencv2/core/utils/logger.hpp"
#include "opencv2/dnn/shape_utils.hpp"

#include "../base/context.h"
#include "../base/tensor.h"
#include "../base/buffer.h"
#include "../base/utility.h"

#include "op_types.h"

#ifdef HAVE_METAL

namespace {

std::string op_softmax_kernel_name(int /* axis, unused */, bool log_softmax)
{
    return log_softmax ? "kernel_log_softmax" : "kernel_softmax";
}

id<MTLBuffer> op_softmax_make_attribute(const cv::dnn::metal::Tensor& src, const cv::dnn::metal::Tensor& dst, int axis, bool log_softmax)
{
    axis = cv::dnn::normalize_axis(axis, src.dims());

    cv::dnn::metal::OpSoftmaxAttribute attr;
    attr.o_size = cv::dnn::metal::shapeCount(src.shape(), 0, axis);
    attr.i_size = cv::dnn::metal::shapeCount(src.shape(), axis + 1);
    attr.r_size = cv::dnn::metal::shapeContent(src.shape(), axis);

    id<MTLBuffer> buffer = [[MTL4DNN_CONTEXT device] newBufferWithBytes:&attr length:sizeof(attr) options:MTLResourceStorageModeShared];
    if (buffer == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": failed to allocate attribute buffer.");
        return nil;
    }
    return buffer;
}

} // anonymous namespace

namespace cv { namespace dnn {namespace metal {

bool OpSoftmax::forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs)
{
    CV_Assert(inputs.size() == 1);
    CV_Assert(outputs.size() == 1);

    Tensor& src = inputs[0];
    Tensor& dst = outputs[0];

    id<MTLComputeCommandEncoder> commandEncoder = [MTL4DNN_CONTEXT makeEncoder];
    if (commandEncoder == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": command encoder is nil.");
        return false;
    }

    NSString* kernelName = [[NSString alloc] initWithFormat:@"%s", op_softmax_kernel_name(axis(), logSoftmax()).c_str()];
    id<MTLComputePipelineState> computePipelineState = [MTL4DNN_CONTEXT findComputePipelineState:kernelName];
    if (computePipelineState == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": compute pipeline state is nil.");
        return false;
    }

    [commandEncoder setComputePipelineState:computePipelineState];
    [commandEncoder setBuffer:src.buffer()->rawBuffer() offset:0 atIndex:0];
    [commandEncoder setBuffer:dst.buffer()->rawBuffer() offset:0 atIndex:1];
    [commandEncoder setBuffer:op_softmax_make_attribute(src, dst, axis(), logSoftmax()) offset:0 atIndex:2];

    // NSUInteger maxTotalThreadsPerThreadgroup = [computePipelineState maxTotalThreadsPerThreadgroup];
    // NSUInteger threadExecutionWidth = [computePipelineState threadExecutionWidth];
    // NSUInteger threadExecutionHeight = maxTotalThreadsPerThreadgroup / threadExecutionWidth;

    MTLSize threads = MTLSizeMake(shapeCount(dst.shape(), axis() + 1), 1, shapeCount(dst.shape(), 0, axis()));
    MTLSize threadsPerThreadgroup = MTLSizeMake(shapeContent(src.shape(), axis()), 1, 1);

    [commandEncoder dispatchThreads:threads threadsPerThreadgroup:threadsPerThreadgroup];

    [commandEncoder endEncoding];

    if (![MTL4DNN_CONTEXT commit])
    {
        CV_LOG_ERROR(NULL, __func__ << ": commit failed");
        return false;
    }

    return true;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
