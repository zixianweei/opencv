#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "op_metal.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
#ifdef HAVE_METAL

bool copyToTensor(metal::Tensor& dst, const Mat& src)
{
    CV_Assert(src.isContinuous() && (src.type() == CV_8S || src.type() == CV_32F));
    return dst.copyDataToDevice(src.data);
}

bool copyToMat(Mat& dst, const metal::Tensor& src)
{
    CV_Assert(dst.isContinuous() && (dst.type() == CV_8S || dst.type() == CV_32F));
    return src.copyDataFromDevice(dst.data);
}

MetalBackendNode::MetalBackendNode(const std::vector<Ptr<BackendWrapper>>& inputsWrapper,
                                   const std::shared_ptr<metal::OpBase>& op,
                                   const std::vector<Ptr<BackendWrapper>>& outputsWrapper)
                                   : BackendNode(DNN_BACKEND_MPS)
{
    operation_ = op;
}

MetalBackendWrapper::MetalBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
    : BackendWrapper(DNN_BACKEND_MPS, DNN_TARGET_METAL)
{
    Ptr<MetalBackendWrapper> base = baseBuffer.dynamicCast<MetalBackendWrapper>();
    CV_Assert(!base.empty());

    host_ = m;
}

void MetalBackendWrapper::copyToHost()
{

}

void MetalBackendWrapper::setHostDirty()
{

}

#endif // HAVE_METAL

void Net::Impl::initMetalBackend()
{
    CV_TRACE_FUNCTION();
    CV_Assert(preferableBackend == DNN_BACKEND_MPS);
    
    if (!haveMetal())
        return;
    
    metal_context_ = metal::Context::create();
    
    for (auto it = layers.begin(); it != layers.end(); it++)
    {
        LayerData& layer_data = it->second;
        if (layer_data.skip)
            continue;
        Ptr<Layer> layer = layer_data.layerInstance;
        if (!layer->supportBackend(preferableBackend))
        {
            std::string msg = "unsupport layer: [" + layer->name + "]";
            CV_LOG_INFO(NULL, msg);
            continue;
        }
            
    }
}

bool haveMetal() {
#ifdef HAVE_METAL
    return true;
#else
    return false;
#endif
}

}}  // namespace cv::dnn
