#include "kernel_DepthScale_alpha_numerator_denominator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    DepthScaleAlphaNumeratorDenominatorKernel(
        float* DepthScale_p_kp1,
        unsigned int DepthScale_p_kp1_num_alloc,
        float* DepthScale_r_k,
        unsigned int DepthScale_r_k_num_alloc,
        float* DepthScale_w,
        unsigned int DepthScale_w_num_alloc,
        float* const DepthScale_total_ag,
        float* const DepthScale_total_ac,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[128];

  __shared__ float DepthScale_total_ag_local[1];

  __shared__ float DepthScale_total_ac_local[1];

  float r0, r1;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(DepthScale_p_kp1,
                                        0 * DepthScale_p_kp1_num_alloc,
                                        global_thread_idx,
                                        r0);
    ReadIdx1<1024, float, float, float>(
        DepthScale_r_k, 0 * DepthScale_r_k_num_alloc, global_thread_idx, r1);
    r1 = r0 * r1;
  };
  SumStore<float>(DepthScale_total_ag_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, float, float, float>(
        DepthScale_w, 0 * DepthScale_w_num_alloc, global_thread_idx, r1);
    r1 = r0 * r1;
  };
  SumStore<float>(DepthScale_total_ac_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  SumFlushFinal<float>(DepthScale_total_ag_local, DepthScale_total_ag, 1);
  SumFlushFinal<float>(DepthScale_total_ac_local, DepthScale_total_ac, 1);
}

void DepthScaleAlphaNumeratorDenominator(
    float* DepthScale_p_kp1,
    unsigned int DepthScale_p_kp1_num_alloc,
    float* DepthScale_r_k,
    unsigned int DepthScale_r_k_num_alloc,
    float* DepthScale_w,
    unsigned int DepthScale_w_num_alloc,
    float* const DepthScale_total_ag,
    float* const DepthScale_total_ac,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  DepthScaleAlphaNumeratorDenominatorKernel<<<n_blocks, 1024>>>(
      DepthScale_p_kp1,
      DepthScale_p_kp1_num_alloc,
      DepthScale_r_k,
      DepthScale_r_k_num_alloc,
      DepthScale_w,
      DepthScale_w_num_alloc,
      DepthScale_total_ag,
      DepthScale_total_ac,
      problem_size);
}

}  // namespace caspar