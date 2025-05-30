#include "op_softmax.h"

#include "opencv2/core/utils/logger.hpp"
#include "opencv2/dnn/shape_utils.hpp"

#include "../base/context.h"
#include "../base/tensor.h"
#include "../base/buffer.h"
#include "../base/utility.h"

#ifdef HAVE_METAL

namespace cv { namespace dnn {namespace metal {

bool OpSoftmax::forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs)
{
    CV_Assert(inputs.size() == 1);
    CV_Assert(outputs.size() == 1);

    Tensor& src = inputs[0];
    Tensor& dst = outputs[0];

#if 0
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
#endif
    id<MTLCommandQueue> commandQueue = [MPS4DNN_CONTEXT commandQueue];
    if (commandQueue == nil)
    {
        CV_LOG_ERROR(NULL, __func__ << ": command queue is nil.");
        return false;
    }

    @autoreleasepool
    {
        MPSGraph* graph = [[MPSGraph alloc] init];
        if (graph == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": mps graph is nil.");
            return false;
        }

        MPSGraphTensor* inputTensor = [graph placeholderWithShape:makeMPSShape(src.shape()) dataType:MPSDataTypeFloat32 name:nil];
        if (inputTensor == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": input tensor is nil.");
            return false;
        }

        MPSGraphTensor* outputTensor = nil;
        if (logSoftmax())
        {
            outputTensor = [graph softMaxWithTensor:inputTensor axis:axis() name:nil];
            outputTensor = [graph logarithmWithTensor:outputTensor name:nil];
        }
        else
        {
            outputTensor = [graph softMaxWithTensor:inputTensor axis:axis() name:nil];
        }
        if (outputTensor == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": output tensor is nil.");
            return false;
        }

        MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:src.buffer()->rawBuffer() shape:makeMPSShape(src.shape()) dataType:MPSDataTypeFloat32];
        if (inputData == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": input data is nil.");
            return false;
        }

        MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:dst.buffer()->rawBuffer() shape:makeMPSShape(dst.shape()) dataType:MPSDataTypeFloat32];
        if (outputData == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": output data is nil.");
            return false;
        }

        MPSGraphTensorDataDictionary* feeds = @{inputTensor : inputData};
        MPSGraphTensorDataDictionary* results = @{outputTensor : outputData};
        MPSGraphExecutionDescriptor* descriptor = [[MPSGraphExecutionDescriptor alloc] init];
        MPSCommandBuffer* commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
        if (commandBuffer == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": command buffer is nil.");
            return false;
        }

        [graph encodeToCommandBuffer:commandBuffer
                               feeds:feeds
                    targetOperations:nil
                   resultsDictionary:results
                 executionDescriptor:descriptor];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    return true;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
