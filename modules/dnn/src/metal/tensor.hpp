#ifndef OPENCV_DNN_METAL_TENSOR_HPP
#define OPENCV_DNN_METAL_TENSOR_HPP

#include <cstdint>
#include <vector>

#ifdef HAVE_METAL
#ifdef __OBJC__
@class TensorImpl;
#else
typedef struct objc_object TensorImpl;
#endif
#endif

namespace cv { namespace dnn { namespace metal {
#ifdef HAVE_METAL

class Tensor {
public:
    Tensor();
    ~Tensor();
    
    bool allocate(size_t size);
    bool allocate(size_t width, size_t height, size_t depth);
    void release();
    
    bool copyDataToDevice(const void* data);
    bool copyDataFromDevice(void* data) const;
    
    size_t size() const;
    std::vector<size_t> shape() const;
    
    void* getBuffer() const;
    void* getTexture() const;

private:
    __strong TensorImpl* impl;
};

#endif // HAVE_METAL
}}}

#endif // !OPENCV_DNN_METAL_TENSOR_HPP
