#include "kernel_PinholeTranslation_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) PinholeTranslationUpdateRFirstKernel(
    float* PinholeTranslation_r_k,
    unsigned int PinholeTranslation_r_k_num_alloc,
    float* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    const float* const negalpha,
    float* out_PinholeTranslation_r_kp1,
    unsigned int out_PinholeTranslation_r_kp1_num_alloc,
    float* const out_PinholeTranslation_r_0_norm2_tot,
    float* const out_PinholeTranslation_r_kp1_norm2_tot,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_PinholeTranslation_r_0_norm2_tot_local[1];

  __shared__ float out_PinholeTranslation_r_kp1_norm2_tot_local[1];

  float r0, r1, r2, r3, r4, r5, r6;

  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_r_k,
                                         0 * PinholeTranslation_r_k_num_alloc,
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
  };
  LoadUnique<1, float, float>(negalpha, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fmaf(r3, r6, r0);
    r4 = fmaf(r4, r6, r1);
    r6 = fmaf(r5, r6, r2);
    WriteIdx3<1024, float, float, float4>(
        out_PinholeTranslation_r_kp1,
        0 * out_PinholeTranslation_r_kp1_num_alloc,
        global_thread_idx,
        r3,
        r4,
        r6);
    r2 = fmaf(r2, r2, r1 * r1);
    r2 = fmaf(r0, r0, r2);
  };
  SumStore<float>(out_PinholeTranslation_r_0_norm2_tot_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r2);
  if (global_thread_idx < problem_size) {
    r4 = fmaf(r4, r4, r6 * r6);
    r4 = fmaf(r3, r3, r4);
  };
  SumStore<float>(out_PinholeTranslation_r_kp1_norm2_tot_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(out_PinholeTranslation_r_0_norm2_tot_local,
                       out_PinholeTranslation_r_0_norm2_tot,
                       1);
  SumFlushFinal<float>(out_PinholeTranslation_r_kp1_norm2_tot_local,
                       out_PinholeTranslation_r_kp1_norm2_tot,
                       1);
}

void PinholeTranslationUpdateRFirst(
    float* PinholeTranslation_r_k,
    unsigned int PinholeTranslation_r_k_num_alloc,
    float* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    const float* const negalpha,
    float* out_PinholeTranslation_r_kp1,
    unsigned int out_PinholeTranslation_r_kp1_num_alloc,
    float* const out_PinholeTranslation_r_0_norm2_tot,
    float* const out_PinholeTranslation_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeTranslationUpdateRFirstKernel<<<n_blocks, 1024>>>(
      PinholeTranslation_r_k,
      PinholeTranslation_r_k_num_alloc,
      PinholeTranslation_w,
      PinholeTranslation_w_num_alloc,
      negalpha,
      out_PinholeTranslation_r_kp1,
      out_PinholeTranslation_r_kp1_num_alloc,
      out_PinholeTranslation_r_0_norm2_tot,
      out_PinholeTranslation_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar