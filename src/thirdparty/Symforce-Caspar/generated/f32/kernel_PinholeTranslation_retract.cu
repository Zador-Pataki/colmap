#include "kernel_PinholeTranslation_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeTranslationRetractKernel(
    float* PinholeTranslation,
    unsigned int PinholeTranslation_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_PinholeTranslation_retracted,
    unsigned int out_PinholeTranslation_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  float r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(PinholeTranslation,
                                         0 * PinholeTranslation_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2);
    ReadIdx3<1024, float, float, float4>(
        delta, 0 * delta_num_alloc, global_thread_idx, r3, r4, r5);
    r3 = r0 + r3;
    r4 = r1 + r4;
    r5 = r2 + r5;
    WriteIdx3<1024, float, float, float4>(
        out_PinholeTranslation_retracted,
        0 * out_PinholeTranslation_retracted_num_alloc,
        global_thread_idx,
        r3,
        r4,
        r5);
  };
}

void PinholeTranslationRetract(
    float* PinholeTranslation,
    unsigned int PinholeTranslation_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_PinholeTranslation_retracted,
    unsigned int out_PinholeTranslation_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeTranslationRetractKernel<<<n_blocks, 1024>>>(
      PinholeTranslation,
      PinholeTranslation_num_alloc,
      delta,
      delta_num_alloc,
      out_PinholeTranslation_retracted,
      out_PinholeTranslation_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar