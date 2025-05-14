#ifndef OPENCV_DNN_METAL_EXEC_OP_TYPES_H
#define OPENCV_DNN_METAL_EXEC_OP_TYPES_H

namespace cv { namespace dnn { namespace metal {

struct OpSoftmaxAttribute
{
    int i_size; // inner size
    int o_size; // outer size
    int r_size; // reduce size
};

}}} // namespace cv::dnn::metal

#endif // !OPENCV_DNN_METAL_EXEC_OP_TYPES_H
