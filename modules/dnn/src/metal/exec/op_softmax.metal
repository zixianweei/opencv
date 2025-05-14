#include <metal_stdlib>

#include "op_types.h"

using namespace metal;

kernel void kernel_softmax(const device float* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           constant cv::dnn::metal::OpSoftmaxAttribute& attr [[buffer(2)]],
                           uint3 gid [[thread_position_in_grid]]) {
  if (any(gid >= uint3(attr.i_size, 1, attr.o_size)))
    return;

  int index = (int)gid.z * attr.i_size * attr.r_size + (int)gid.x;

  float maxv = -INFINITY;
  for (int ri = 0; ri < attr.r_size; ri++) {
      maxv = max(maxv, src[index + ri * attr.i_size]);
  }

  float sum = 0.F;
  for (int ri = 0; ri < attr.r_size; ri++) {
    float v = exp(float(src[index + ri * attr.i_size] - maxv));
    sum += v;
    dst[index + ri * attr.i_size] = v;
  }

  for (int ri = 0; ri < attr.r_size; ri++) {
    int i = index + ri * attr.i_size;
    dst[i] /= sum;
  }
}

kernel void kernel_log_softmax(const device float* src [[buffer(0)]],
                               device float* dst [[buffer(1)]],
                               constant cv::dnn::metal::OpSoftmaxAttribute& attr [[buffer(2)]],
                               uint3 gid [[thread_position_in_grid]]) {
  if (any(gid >= uint3(attr.i_size, 1, attr.o_size)))
    return;

  int index = (int)gid.z * attr.i_size * attr.r_size + (int)gid.x;

  float maxv = -INFINITY;
  for (int ri = 0; ri < attr.r_size; ri++) {
      maxv = max(maxv, src[index + ri * attr.i_size]);
  }

  float sum = 0.F;
  for (int ri = 0; ri < attr.r_size; ri++) {
    float v = exp(float(src[index + ri * attr.i_size] - maxv));
    sum += v;
    dst[index + ri * attr.i_size] = v;
  }

  for (int ri = 0; ri < attr.r_size; ri++) {
    int i = index + ri * attr.i_size;
    dst[i] /= sum;
    dst[i] = log(dst[i]);
  }
}
