#include "op_conv.h"

#include "opencv2/core/utils/logger.hpp"
#include "opencv2/dnn/shape_utils.hpp"

#include "../base/context.h"
#include "../base/tensor.h"
#include "../base/buffer.h"
#include "../base/utility.h"
#include "../../op_metal.hpp"

#ifdef HAVE_METAL

namespace cv { namespace dnn {namespace metal {

bool OpConv::forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs)
{
    CV_Assert(inputs.size() == 1);
    CV_Assert(outputs.size() == 1);

    Tensor& src = inputs[0];
    Tensor& dst = outputs[0];

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

        MPSGraphTensor* inputTensor = [graph placeholderWithShape:makeMPSShape(inputs[0].shape()) dataType:MPSDataTypeFloat32 name:nil];
        if (inputTensor == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": input tensor is nil.");
            return false;
        }

        MPSGraphTensor* weightsTensor = [graph placeholderWithShape:makeMPSShape(cv::dnn::shape(weights_)) dataType:MPSDataTypeFloat32 name:nil];
        if (weightsTensor == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": weights tensor is nil.");
            return false;
        }

        MPSGraphConvolution2DOpDescriptor* descriptor = [[MPSGraphConvolution2DOpDescriptor alloc] autorelease];
        descriptor.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
        descriptor.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
        descriptor.groups = nGroups_;
        descriptor.paddingStyle = MPSGraphPaddingStyleExplicit;
        descriptor.strideInX = strideInX_;
        descriptor.strideInY = strideInY_;
        descriptor.dilationRateInX = dilationRateInX_;
        descriptor.dilationRateInY = dilationRateInY_;
        descriptor.paddingTop = paddingTop_;
        descriptor.paddingLeft = paddingLeft_;
        descriptor.paddingBottom = paddingBottom_;
        descriptor.paddingRight = paddingRight_;

        MPSGraphTensor* outputTensor = [graph convolution2DWithSourceTensor:inputTensor weightsTensor:weightsTensor descriptor:descriptor name:nil];
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

        Tensor weightsDataTensor(weights_.data, shape(weights_));
        MPSGraphTensorData* weightsData = [[MPSGraphTensorData alloc] initWithMTLBuffer:weightsDataTensor.buffer()->rawBuffer() shape:makeMPSShape(weightsDataTensor.shape()) dataType:MPSDataTypeFloat32];
        if (weightsData == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": weights data is nil.");
            return false;
        }

        MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:dst.buffer()->rawBuffer() shape:makeMPSShape(dst.shape()) dataType:MPSDataTypeFloat32];
        if (outputData == nil)
        {
            CV_LOG_ERROR(NULL, __func__ << ": output data is nil.");
            return false;
        }

        MPSGraphTensorDataDictionary* feeds = @{
            inputTensor: inputData,
            weightsTensor: weightsData
        };
        MPSGraphTensorDataDictionary* results = @{
            outputTensor: outputData
        };
        MPSGraphExecutionDescriptor* executionDescriptor = [[MPSGraphExecutionDescriptor alloc] init];
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
                 executionDescriptor:executionDescriptor];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }

    return true;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
