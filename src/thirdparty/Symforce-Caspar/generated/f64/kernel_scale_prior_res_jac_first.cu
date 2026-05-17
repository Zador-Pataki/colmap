#include "kernel_scale_prior_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ScalePriorResJacFirstKernel(double* scale,
                                unsigned int scale_num_alloc,
                                SharedIndex* scale_indices,
                                double* inv_std,
                                unsigned int inv_std_num_alloc,
                                double* loss,
                                unsigned int loss_num_alloc,
                                double* out_res,
                                unsigned int out_res_num_alloc,
                                double* const out_rTr,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;

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
    r5 = r4 * r4;
    ReadIdx1<1024, double, double, double>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r6);
    r7 = r6 * r6;
    r7 = r5 * r7;
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r5, r8);
    r9 = 5.00000000000000000e-01;
    r8 = fmax(r8, r0);
    r10 = r8 * r8;
    r11 = -1.00000000000000000e+00;
    r12 = 2.00000000000000000e+00;
    r13 = r12 * r8;
    r14 = fmax(r0, r7);
    r15 = sqrt(r14);
    r13 = fma(r15, r13, r11 * r10);
    r13 = r7 <= r10 ? r7 : r13;
    r15 = 2.50000000000000000e+00;
    r16 = 1.00000000000000000e+00;
    r17 = 1.0 / r10;
    r17 = fma(r17, r7, r16);
    r18 = log(r17);
    r18 = r18 * r10;
    r13 = r5 < r15 ? r18 : r13;
    r18 = 1.50000000000000000e+00;
    r19 = sqrt(r17);
    r19 = r11 + r19;
    r19 = r12 * r19;
    r19 = r19 * r10;
    r13 = r5 < r18 ? r19 : r13;
    r13 = r5 < r9 ? r7 : r13;
    r19 = fmax(r2, r13);
    r20 = 1.0 / r14;
    r20 = r1 * r20;
    r21 = r19 * r20;
    r22 = sqrt(r21);
    r22 = r7 <= r0 ? r3 : r22;
    r4 = r4 * r6;
    r3 = r22 * r4;
    WriteIdx1<1024, double, double, double>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r3);
    r3 = r22 * r22;
    r3 = r7 * r3;
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  if (global_thread_idx < problem_size) {
    r3 = r11 * r22;
    r23 = r6 * r4;
    r24 = r12 * r23;
    r25 = -1.00000000000000008e-15;
    r25 = r7 + r25;
    r25 = copysign(1.0, r25);
    r25 = r16 + r25;
    r26 = r8 * r25;
    r27 = rsqrt(r14);
    r26 = r26 * r27;
    r26 = r26 * r23;
    r26 = r7 <= r10 ? r24 : r26;
    r27 = 1.0 / r17;
    r27 = r12 * r27;
    r27 = r27 * r23;
    r26 = r5 < r15 ? r27 : r26;
    r17 = rsqrt(r17);
    r17 = r12 * r17;
    r17 = r17 * r23;
    r26 = r5 < r18 ? r17 : r26;
    r26 = r5 < r9 ? r24 : r26;
    r24 = r9 * r26;
    r13 = copysign(1.0, r13);
    r13 = r16 + r13;
    r20 = r13 * r20;
    r13 = r1 * r11;
    r14 = r14 * r14;
    r14 = 1.0 / r14;
    r13 = r13 * r19;
    r13 = r13 * r25;
    r13 = r13 * r14;
    r13 = fma(r23, r13, r20 * r24);
    r13 = r9 * r13;
    r21 = rsqrt(r21);
    r13 = r13 * r21;
    r13 = r7 <= r0 ? r2 : r13;
    r13 = fma(r13, r4, r6 * r22);
    r22 = 2.50000000000000000e-01;
    r10 = r7 <= r10 ? r2 : r2;
    r10 = r5 < r15 ? r2 : r10;
    r10 = r5 < r18 ? r2 : r10;
    r10 = r5 < r9 ? r2 : r10;
    r10 = r22 * r10;
    r10 = r10 * r21;
    r10 = r10 * r20;
    r10 = r7 <= r0 ? r2 : r10;
    r2 = r11 * r10;
    r13 = fma(r4, r2, r13);
    r3 = r3 * r13;
    r3 = r3 * r4;
    WriteSum1<double, double>((double*)inout_shared, r3);
  };
  FlushSumShared<1, double>(out_scale_njtr,
                            0 * out_scale_njtr_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r13 * r13;
    WriteSum1<double, double>((double*)inout_shared, r13);
  };
  FlushSumShared<1, double>(out_scale_precond_diag,
                            0 * out_scale_precond_diag_num_alloc,
                            scale_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ScalePriorResJacFirst(double* scale,
                           unsigned int scale_num_alloc,
                           SharedIndex* scale_indices,
                           double* inv_std,
                           unsigned int inv_std_num_alloc,
                           double* loss,
                           unsigned int loss_num_alloc,
                           double* out_res,
                           unsigned int out_res_num_alloc,
                           double* const out_rTr,
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
  ScalePriorResJacFirstKernel<<<n_blocks, 1024>>>(
      scale,
      scale_num_alloc,
      scale_indices,
      inv_std,
      inv_std_num_alloc,
      loss,
      loss_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_scale_njtr,
      out_scale_njtr_num_alloc,
      out_scale_precond_diag,
      out_scale_precond_diag_num_alloc,
      out_scale_precond_tril,
      out_scale_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar