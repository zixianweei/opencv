#include "op_matmul.h"

#ifdef HAVE_METAL

namespace cv { namespace dnn {namespace metal {

bool OpMatMul::forward(std::vector<Tensor>& inputs, std::vector<Tensor>& outputs)
{

    return true;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
