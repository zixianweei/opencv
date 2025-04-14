#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "op_metal.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
#ifdef HAVE_METAL

bool copyToTensor(metal::Tensor& dst, const Mat& src)
{
    CV_Assert(src.isContinuous() && (src.type() == CV_8S || src.type() == CV_32F));
    std::vector<int> shape = cv::dnn::shape(src);
    return dst.reshape(src.data, shape);
}

bool copyToMat(Mat& dst, const metal::Tensor& src)
{
    CV_Assert(dst.isContinuous() && (dst.type() == CV_8S || dst.type() == CV_32F));
    return src.copyDataFromDevice(dst.data);
}

void createTensors(const std::vector<Ptr<BackendWrapper>>& wrappers, std::vector<metal::Tensor>& tensors)
{
    tensors.reserve(wrappers.size());
    for (const Ptr<BackendWrapper>& wrapper : wrappers)
    {
        CV_Assert(!wrapper.empty());
        tensors.push_back(wrapper.dynamicCast<MetalBackendWrapper>()->getTensor());
    }
}

MetalBackendNode::MetalBackendNode(const std::vector<Ptr<BackendWrapper>>& inputs_wrapper,
                                   const std::shared_ptr<metal::OpBase>& op,
                                   const std::vector<Ptr<BackendWrapper>>& outputs_wrapper)
                                   : BackendNode(DNN_BACKEND_MPS)
{
    operation_ = op;
    
    inputs_wrapper_ = inputs_wrapper;
    createTensors(inputs_wrapper_, inputs_);
    
    outputs_wrapper_ = outputs_wrapper;
    createTensors(outputs_wrapper_, outputs_);
}

bool MetalBackendNode::forward()
{
    for (auto& e : inputs_wrapper_)
    {
        e.dynamicCast<MetalBackendWrapper>()->copyToDevice();
    }
    return operation_->forward(inputs_, outputs_);
}

MetalBackendWrapper::MetalBackendWrapper(Mat& m)
: BackendWrapper(DNN_BACKEND_MPS, DNN_TARGET_METAL)
{
    host_ = m;
}

MetalBackendWrapper::MetalBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
    : BackendWrapper(DNN_BACKEND_MPS, DNN_TARGET_METAL)
{
    Ptr<MetalBackendWrapper> base = baseBuffer.dynamicCast<MetalBackendWrapper>();
    CV_Assert(!base.empty());

    host_ = m;
    
    setHostDirty();
    
    this->copyToDevice();
}

void MetalBackendWrapper::copyToHost()
{

}

void MetalBackendWrapper::setHostDirty()
{
    host_dirty_ = true;
}

void MetalBackendWrapper::copyToDevice()
{
    if (host_dirty_)
    {
        copyToTensor(tensor_, host_);
        host_dirty_ = false;
    }
}

metal::Tensor MetalBackendWrapper::getTensor()
{
    return tensor_;
}

#endif // HAVE_METAL

void Net::Impl::initMetalBackend()
{
    CV_TRACE_FUNCTION();
    CV_Assert(preferableBackend == DNN_BACKEND_MPS);
    
    if (!haveMetal())
        return;
    
    for (auto it = layers.begin(); it != layers.end(); it++)
    {
        LayerData& layer_data = it->second;
        if (layer_data.skip)
            continue;
        Ptr<Layer> layer = layer_data.layerInstance;
        if (!layer->supportBackend(preferableBackend))
        {
            continue;
        }
        
        // TODO(zixianwei): Remove this when push to mainstream. For debug purpose only.
        CV_LOG_INFO(NULL, "layer: [" + layer->name + "]");
        
        try
        {
            layer_data.backendNodes[DNN_BACKEND_MPS] = layer->initMetal(layer_data.inputBlobsWrappers, layer_data.outputBlobsWrappers);
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "initMetal failed, fallback to CPU implementation. " << e.what());
            layer_data.backendNodes[DNN_BACKEND_MPS] = Ptr<BackendNode>();
        }
    }
}

void forwardMetal(std::vector<Ptr<BackendWrapper> > &outputs, const Ptr<BackendNode>& node)
{
#ifdef HAVE_METAL
    CV_Assert(!node.empty());
    
    Ptr<MetalBackendNode> metal_node = node.dynamicCast<MetalBackendNode>();
    
    CV_Assert(metal_node->forward());
    // TODO(zixianwei): setDirty
#endif
}

bool haveMetal() {
#ifdef HAVE_METAL
    return metal::isAvailable();
#else
    return false;
#endif
}

}}  // namespace cv::dnn
