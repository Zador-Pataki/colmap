#include "kernel_PinholeTranslation_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeTranslationAlphaDenominatorOrBetaNumeratorKernel(
        float* PinholeTranslation_p_kp1,
        unsigned int PinholeTranslation_p_kp1_num_alloc,
        float* PinholeTranslation_w,
        unsigned int PinholeTranslation_w_num_alloc,
        float* const PinholeTranslation_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float PinholeTranslation_out_local[1];

  float r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_p_kp1,
                                         0 * PinholeTranslation_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2);
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_w,
                                         0 * PinholeTranslation_w_num_alloc,
                                         global_thread_idx,
                                         r3,
                                         r4,
                                         r5);
    r4 = fmaf(r1, r4, r2 * r5);
    r4 = fmaf(r0, r3, r4);
  };
  SumStore<float>(PinholeTranslation_out_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(PinholeTranslation_out_local, PinholeTranslation_out, 1);
}

void PinholeTranslationAlphaDenominatorOrBetaNumerator(
    float* PinholeTranslation_p_kp1,
    unsigned int PinholeTranslation_p_kp1_num_alloc,
    float* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    float* const PinholeTranslation_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeTranslationAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      PinholeTranslation_p_kp1,
      PinholeTranslation_p_kp1_num_alloc,
      PinholeTranslation_w,
      PinholeTranslation_w_num_alloc,
      PinholeTranslation_out,
      problem_size);
}

}  // namespace caspar