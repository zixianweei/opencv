#include "op_softmax.hpp"

namespace cv { namespace dnn {namespace metal {
#ifdef HAVE_METAL

bool OpSoftmax::forward(std::vector<Tensor> &ins, std::vector<Tensor> &outs)
{
    return false;
}

#endif // HAVE_METAL
}}}
