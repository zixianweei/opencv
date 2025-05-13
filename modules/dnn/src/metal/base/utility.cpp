#include "utility.h"

#include "opencv2/core/utils/logger.hpp"

#ifdef HAVE_METAL

namespace cv { namespace dnn { namespace metal {

int shapeContent(const MatShape &shape, int axis) {
    return shape.size() > axis ? shape[axis] : 1;
}

int shapeCount(const MatShape &shape, int start_axis, int end_axis)
{
    if (-1 == end_axis || end_axis > shape.size())
    {
        end_axis = static_cast<int>(shape.size());
    }

    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
        count *= shape[i];
    }
    return count;
}

int elementSize(Format format)
{
    switch (format)
    {
    case Format::kUnsignedChar8:
        return 1;
    case Format::kFloat32:
        return 4;
    default:
        break;
    }
    CV_LOG_FATAL(NULL, __func__ << ": elementSize unreachable.");
    return 0;
}

}}} // namespace cv::dnn::metal

#endif // HAVE_METAL
