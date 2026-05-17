#include "kernel_scale_prior_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ScalePriorResJacKernel(double* scale,
                           unsigned int scale_num_alloc,
                           SharedIndex* scale_indices,
                           double* inv_std,
                           unsigned int inv_std_num_alloc,
                           double* loss,
                           unsigned int loss_num_alloc,
                           double* out_res,
                           unsigned int out_res_num_alloc,
                           double* const out_scale_njtr,
                           unsigned int out_scale_njtr_num_alloc,
                           double* const out_scale_precond_diag,
                           unsigned int out_scale_precond_diag_num_alloc,
                           double* const out_scale_precond_tril,
                           unsigned int out_scale_precond_tril_num_alloc,
                           size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29;

  if (global_thread_idx < problem_size) {
    r0 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r1);
    r2 = 0.00000000000000000e+00;
    r1 = fmax(r1, r2);
    r3 = sqrt(r1);
  };
  LoadShared<1, double, double>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, scale_indices_loc[threadIdx.x].target, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r5);
    r6 = r4 * r5;
    r7 = r5 * r6;
    r8 = r4 * r7;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r9, r10);
    r11 = 5.00000000000000000e-01;
    r10 = fmax(r10, r0);
    r12 = r10 * r10;
    r13 = -1.00000000000000000e+00;
    r14 = 2.00000000000000000e+00;
    r15 = r14 * r10;
    r16 = fmax(r0, r8);
    r17 = sqrt(r16);
    r15 = fma(r17, r15, r13 * r12);
    r15 = r8 <= r12 ? r8 : r15;
    r17 = 2.50000000000000000e+00;
    r18 = 1.00000000000000000e+00;
    r19 = 1.0 / r12;
    r20 = r4 * r19;
    r20 = fma(r7, r20, r18);
    r21 = log(r20);
    r21 = r21 * r12;
    r15 = r9 < r17 ? r21 : r15;
    r21 = 1.50000000000000000e+00;
    r22 = sqrt(r20);
    r22 = r13 + r22;
    r22 = r14 * r22;
    r22 = r22 * r12;
    r15 = r9 < r21 ? r22 : r15;
    r15 = r9 < r11 ? r8 : r15;
    r22 = fmax(r2, r15);
    r23 = 1.0 / r16;
    r23 = r1 * r23;
    r24 = r22 * r23;
    r25 = sqrt(r24);
    r25 = r8 <= r0 ? r3 : r25;
    r3 = r25 * r6;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3);
    r3 = r13 * r25;
    r26 = r4 * r14;
    r27 = r5 * r5;
    r26 = r26 * r27;
    r27 = -1.00000000000000008e-15;
    r27 = r27 + r8;
    r27 = copysign(1.0, r27);
    r27 = r18 + r27;
    r28 = r10 * r27;
    r29 = rsqrt(r16);
    r28 = r28 * r29;
    r28 = r28 * r7;
    r28 = r8 <= r12 ? r26 : r28;
    r29 = 1.0 / r20;
    r29 = r29 * r26;
    r28 = r9 < r17 ? r29 : r28;
    r20 = rsqrt(r20);
    r20 = r20 * r26;
    r28 = r9 < r21 ? r20 : r28;
    r28 = r9 < r11 ? r26 : r28;
    r26 = r11 * r28;
    r15 = copysign(1.0, r15);
    r15 = r18 + r15;
    r23 = r15 * r23;
    r15 = r1 * r13;
    r16 = r16 * r16;
    r16 = 1.0 / r16;
    r15 = r15 * r22;
    r15 = r15 * r27;
    r15 = r15 * r16;
    r15 = fma(r7, r15, r23 * r26);
    r15 = r11 * r15;
    r24 = rsqrt(r24);
    r15 = r15 * r24;
    r15 = r8 <= r0 ? r2 : r15;
    r15 = fma(r15, r6, r5 * r25);
    r25 = 2.50000000000000000e-01;
    r12 = r8 <= r12 ? r2 : r2;
    r12 = r9 < r17 ? r2 : r12;
    r12 = r9 < r21 ? r2 : r12;
    r12 = r9 < r11 ? r2 : r12;
    r12 = r25 * r12;
    r12 = r12 * r24;
    r12 = r12 * r23;
    r12 = r8 <= r0 ? r2 : r12;
    r2 = r13 * r12;
    r15 = fma(r6, r2, r15);
    r3 = r3 * r15;
    r3 = r3 * r6;
    WriteSum1<double, double>((double*)inout_shared, r3);
  };
  FlushSumShared<1, double>(out_scale_njtr,
                            0 * out_scale_njtr_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = r15 * r15;
    WriteSum1<double, double>((double*)inout_shared, r15);
  };
  FlushSumShared<1, double>(out_scale_precond_diag,
                            0 * out_scale_precond_diag_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
}

void ScalePriorResJac(double* scale,
                      unsigned int scale_num_alloc,
                      SharedIndex* scale_indices,
                      double* inv_std,
                      unsigned int inv_std_num_alloc,
                      double* loss,
                      unsigned int loss_num_alloc,
                      double* out_res,
                      unsigned int out_res_num_alloc,
                      double* const out_scale_njtr,
                      unsigned int out_scale_njtr_num_alloc,
                      double* const out_scale_precond_diag,
                      unsigned int out_scale_precond_diag_num_alloc,
                      double* const out_scale_precond_tril,
                      unsigned int out_scale_precond_tril_num_alloc,
                      size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ScalePriorResJacKernel<<<n_blocks, 1024>>>(scale,
                                             scale_num_alloc,
                                             scale_indices,
                                             inv_std,
                                             inv_std_num_alloc,
                                             loss,
                                             loss_num_alloc,
                                             out_res,
                                             out_res_num_alloc,
                                             out_scale_njtr,
                                             out_scale_njtr_num_alloc,
                                             out_scale_precond_diag,
                                             out_scale_precond_diag_num_alloc,
                                             out_scale_precond_tril,
                                             out_scale_precond_tril_num_alloc,
                                             problem_size);
}

}  // namespace caspar