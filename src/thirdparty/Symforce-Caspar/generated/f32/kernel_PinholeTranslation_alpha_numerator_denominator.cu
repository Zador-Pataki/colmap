#include "kernel_PinholeTranslation_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeTranslationAlphaNumeratorDenominatorKernel(
        float* PinholeTranslation_p_kp1,
        unsigned int PinholeTranslation_p_kp1_num_alloc,
        float* PinholeTranslation_r_k,
        unsigned int PinholeTranslation_r_k_num_alloc,
        float* PinholeTranslation_w,
        unsigned int PinholeTranslation_w_num_alloc,
        float* const PinholeTranslation_total_ag,
        float* const PinholeTranslation_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float PinholeTranslation_total_ag_local[1];

  __shared__ float PinholeTranslation_total_ac_local[1];

  float r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_p_kp1,
                                         0 * PinholeTranslation_p_kp1_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2);
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_r_k,
                                         0 * PinholeTranslation_r_k_num_alloc,
                                         global_thread_idx,
                                         r3,
                                         r4,
                                         r5);
    r3 = fmaf(r0, r3, r1 * r4);
    r3 = fmaf(r2, r5, r3);
  };
  SumStore<float>(PinholeTranslation_total_ag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r3);
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_w,
                                         0 * PinholeTranslation_w_num_alloc,
                                         global_thread_idx,
                                         r3,
                                         r5,
                                         r4);
    r5 = fmaf(r1, r5, r2 * r4);
    r5 = fmaf(r0, r3, r5);
  };
  SumStore<float>(PinholeTranslation_total_ac_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r5);
  SumFlushFinal<float>(
      PinholeTranslation_total_ag_local, PinholeTranslation_total_ag, 1);
  SumFlushFinal<float>(
      PinholeTranslation_total_ac_local, PinholeTranslation_total_ac, 1);
}

void PinholeTranslationAlphaNumeratorDenominator(
    float* PinholeTranslation_p_kp1,
    unsigned int PinholeTranslation_p_kp1_num_alloc,
    float* PinholeTranslation_r_k,
    unsigned int PinholeTranslation_r_k_num_alloc,
    float* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    float* const PinholeTranslation_total_ag,
    float* const PinholeTranslation_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeTranslationAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      PinholeTranslation_p_kp1,
      PinholeTranslation_p_kp1_num_alloc,
      PinholeTranslation_r_k,
      PinholeTranslation_r_k_num_alloc,
      PinholeTranslation_w,
      PinholeTranslation_w_num_alloc,
      PinholeTranslation_total_ag,
      PinholeTranslation_total_ac,
      problem_size);
}

}  // namespace caspar