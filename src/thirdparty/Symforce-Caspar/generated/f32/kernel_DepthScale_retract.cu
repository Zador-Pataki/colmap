#include "kernel_DepthScale_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    DepthScaleRetractKernel(float* DepthScale,
                            unsigned int DepthScale_num_alloc,
                            float* delta,
                            unsigned int delta_num_alloc,
                            float* out_DepthScale_retracted,
                            unsigned int out_DepthScale_retracted_num_alloc,
                            size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(
        DepthScale, 0 * DepthScale_num_alloc, global_thread_idx, r0);
    ReadIdx1<1024, float, float, float>(
        delta, 0 * delta_num_alloc, global_thread_idx, r1);
    r1 = r0 + r1;
    WriteIdx1<1024, float, float, float>(out_DepthScale_retracted,
                                         0 * out_DepthScale_retracted_num_alloc,
                                         global_thread_idx,
                                         r1);
  };
}

void DepthScaleRetract(float* DepthScale,
                       unsigned int DepthScale_num_alloc,
                       float* delta,
                       unsigned int delta_num_alloc,
                       float* out_DepthScale_retracted,
                       unsigned int out_DepthScale_retracted_num_alloc,
                       size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  DepthScaleRetractKernel<<<n_blocks, 1024>>>(
      DepthScale,
      DepthScale_num_alloc,
      delta,
      delta_num_alloc,
      out_DepthScale_retracted,
      out_DepthScale_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar