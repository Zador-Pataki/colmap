#include "kernel_PinholeCalib_normalize.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeCalibNormalizeKernel(float* precond_diag,
                                unsigned int precond_diag_num_alloc,
                                float* precond_tril,
                                unsigned int precond_tril_num_alloc,
                                float* njtr,
                                unsigned int njtr_num_alloc,
                                const float* const diag,
                                float* out_normalized,
                                unsigned int out_normalized_num_alloc,
                                size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[4096];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17;
  LoadUnique<1, float, float>(diag, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<float>((float*)inout_shared, 0, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = 9.99999999999999955e-07;
    r1 = r0 * r1;
    r2 = -1.00000000000000000e+00;
    ReadIdx2<1024, float, float, float2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r3, r4);
    ReadIdx4<1024, float, float, float4>(precond_tril,
                                         0 * precond_tril_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7,
                                         r8);
    r9 = r6 * r7;
    ReadIdx4<1024, float, float, float4>(precond_diag,
                                         0 * precond_diag_num_alloc,
                                         global_thread_idx,
                                         r10,
                                         r11,
                                         r12,
                                         r13);
    r14 = 1.00000000000000000e+00;
    r14 = r0 + r14;
    r10 = fmaf(r10, r14, r1);
    r10 = 1.0 / r10;
    r0 = r2 * r10;
    r15 = r5 * r0;
    r8 = fmaf(r6, r15, r8);
    r11 = fmaf(r11, r14, r1);
    r11 = fmaf(r5, r15, r11);
    r11 = 1.0 / r11;
    r5 = r8 * r11;
    r3 = fmaf(r7, r15, r3);
    r9 = fmaf(r3, r5, r10 * r9);
    r9 = fmaf(r2, r9, r4);
    r12 = fmaf(r12, r14, r1);
    r4 = r6 * r6;
    r4 = fmaf(r10, r4, r8 * r5);
    r12 = fmaf(r2, r4, r12);
    r12 = 1.0 / r12;
    r4 = r9 * r12;
    r8 = r3 * r3;
    r8 = fmaf(r11, r8, r9 * r4);
    r9 = r7 * r7;
    r8 = fmaf(r10, r9, r8);
    r8 = fmaf(r2, r8, r1);
    r8 = fmaf(r13, r14, r8);
    r8 = 1.0 / r8;
    ReadIdx4<1024, float, float, float4>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r14, r13, r1, r9);
    r13 = fmaf(r14, r15, r13);
    r16 = r2 * r13;
    r16 = r16 * r3;
    r16 = fmaf(r11, r16, r9);
    r9 = r7 * r14;
    r16 = fmaf(r0, r9, r16);
    r5 = r2 * r5;
    r1 = fmaf(r13, r5, r1);
    r17 = r6 * r14;
    r1 = fmaf(r0, r17, r1);
    r17 = r2 * r1;
    r16 = fmaf(r4, r17, r16);
    r16 = r8 * r16;
    r8 = r2 * r16;
    r12 = fmaf(r1, r12, r4 * r8);
    r5 = fmaf(r12, r5, r13 * r11);
    r8 = r2 * r3;
    r8 = r8 * r11;
    r5 = fmaf(r16, r8, r5);
    r15 = fmaf(r5, r15, r14 * r10);
    r10 = r7 * r0;
    r15 = fmaf(r16, r10, r15);
    r8 = r6 * r12;
    r15 = fmaf(r0, r8, r15);
    WriteIdx4<1024, float, float, float4>(out_normalized,
                                          0 * out_normalized_num_alloc,
                                          global_thread_idx,
                                          r15,
                                          r5,
                                          r12,
                                          r16);
  };
}

void PinholeCalibNormalize(float* precond_diag,
                           unsigned int precond_diag_num_alloc,
                           float* precond_tril,
                           unsigned int precond_tril_num_alloc,
                           float* njtr,
                           unsigned int njtr_num_alloc,
                           const float* const diag,
                           float* out_normalized,
                           unsigned int out_normalized_num_alloc,
                           size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeCalibNormalizeKernel<<<n_blocks, 1024>>>(precond_diag,
                                                  precond_diag_num_alloc,
                                                  precond_tril,
                                                  precond_tril_num_alloc,
                                                  njtr,
                                                  njtr_num_alloc,
                                                  diag,
                                                  out_normalized,
                                                  out_normalized_num_alloc,
                                                  problem_size);
}

}  // namespace caspar