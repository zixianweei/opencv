#ifndef OPENCV_DNN_OP_METAL_HPP
#define OPENCV_DNN_OP_METAL_HPP

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_METAL
#include "metal/metal.hpp"
#endif

namespace cv { namespace dnn {
#ifdef HAVE_METAL

bool copyToTensor(metal::Tensor& dst, const Mat& src);
bool copyToMat(Mat& dst, const metal::Tensor& src);

class MetalBackendNode : public BackendNode
{
public:
    MetalBackendNode(const std::vector<Ptr<BackendWrapper>>& inputs_wrapper,
                     const std::shared_ptr<metal::OpBase>& op,
                     const std::vector<Ptr<BackendWrapper>>& outputs_wrapper);
    bool forward();

private:
    std::vector<metal::Tensor> inputs_;
    std::vector<metal::Tensor> outputs_;
    std::vector<Ptr<BackendWrapper>> inputs_wrapper_;
    std::vector<Ptr<BackendWrapper>> outputs_wrapper_;
    Ptr<metal::OpBase> operation_;
};

class MetalBackendWrapper : public BackendWrapper
{
public:
    MetalBackendWrapper(Mat& m);
    MetalBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m);

    virtual void copyToHost() CV_OVERRIDE;
    virtual void setHostDirty() CV_OVERRIDE;
    
    void copyToDevice();
    
    metal::Tensor getTensor();

private:
    Mat host_;
    metal::Tensor tensor_;
    bool host_dirty_;
    bool device_dirty_;
};

#endif // HAVE_METAL

void forwardMetal(std::vector<Ptr<BackendWrapper> > &outputs, const Ptr<BackendNode>& node);

bool haveMetal();

}}  // namespace cv::dnn

#endif // !OPENCV_DNN_OP_METAL_HPP
