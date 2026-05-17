#include "kernel_PinholeTranslation_pred_decrease_times_two.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeTranslationPredDecreaseTimesTwoKernel(
        float* PinholeTranslation_step,
        unsigned int PinholeTranslation_step_num_alloc,
        float* PinholeTranslation_precond_diag,
        unsigned int PinholeTranslation_precond_diag_num_alloc,
        const float* const diag,
        float* PinholeTranslation_njtr,
        unsigned int PinholeTranslation_njtr_num_alloc,
        float* const out_PinholeTranslation_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  __shared__ float out_PinholeTranslation_pred_dec_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10;

  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_step,
                                         0 * PinholeTranslation_step_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2);
    ReadIdx3<1024, float, float, float4>(PinholeTranslation_njtr,
                                         0 * PinholeTranslation_njtr_num_alloc,
                                         global_thread_idx,
                                         r3,
                                         r4,
                                         r5);
    ReadIdx3<1024, float, float, float4>(
        PinholeTranslation_precond_diag,
        0 * PinholeTranslation_precond_diag_num_alloc,
        global_thread_idx,
        r6,
        r7,
        r8);
    r9 = r1 * r7;
  };
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r10, r9, r4);
    r4 = r0 * r6;
    r4 = fmaf(r10, r4, r3);
    r4 = fmaf(r0, r4, r1 * r9);
    r9 = r2 * r8;
    r9 = fmaf(r10, r9, r5);
    r4 = fmaf(r2, r9, r4);
  };
  SumStore<float>(out_PinholeTranslation_pred_dec_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r4);
  SumFlushFinal<float>(out_PinholeTranslation_pred_dec_local,
                       out_PinholeTranslation_pred_dec,
                       1);
}

void PinholeTranslationPredDecreaseTimesTwo(
    float* PinholeTranslation_step,
    unsigned int PinholeTranslation_step_num_alloc,
    float* PinholeTranslation_precond_diag,
    unsigned int PinholeTranslation_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeTranslation_njtr,
    unsigned int PinholeTranslation_njtr_num_alloc,
    float* const out_PinholeTranslation_pred_dec,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeTranslationPredDecreaseTimesTwoKernel<<<n_blocks, 1024>>>(
      PinholeTranslation_step,
      PinholeTranslation_step_num_alloc,
      PinholeTranslation_precond_diag,
      PinholeTranslation_precond_diag_num_alloc,
      diag,
      PinholeTranslation_njtr,
      PinholeTranslation_njtr_num_alloc,
      out_PinholeTranslation_pred_dec,
      problem_size);
}

}  // namespace caspar