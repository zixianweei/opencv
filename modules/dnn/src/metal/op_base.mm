#include "op_base.hpp"

#include "context.hpp"

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

OpBase::OpBase(std::string name) : name_(std::move(name))
{
//    auto devName = [Context::getInstance().device() name];
}

#endif // HAVE_METAL
}}}
