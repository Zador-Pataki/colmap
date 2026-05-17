#include "kernel_scale_prior_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ScalePriorScoreKernel(double* scale,
                          unsigned int scale_num_alloc,
                          SharedIndex* scale_indices,
                          double* inv_std,
                          unsigned int inv_std_num_alloc,
                          double* loss,
                          unsigned int loss_num_alloc,
                          double* const out_rTr,
                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ SharedIndex scale_indices_loc[1024];
  scale_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? scale_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  LoadShared<1, double, double>(
      scale, 0 * scale_num_alloc, scale_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, scale_indices_loc[threadIdx.x].target, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r0;
    ReadIdx1<1024, double, double, double>(
        inv_std, 0 * inv_std_num_alloc, global_thread_idx, r1);
    r1 = r1 * r1;
    r1 = r0 * r1;
    r0 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        loss, 2 * loss_num_alloc, global_thread_idx, r2);
    r3 = 0.00000000000000000e+00;
    r2 = fmax(r2, r3);
    r4 = sqrt(r2);
    ReadIdx2<1024, double, double, double2>(
        loss, 0 * loss_num_alloc, global_thread_idx, r5, r6);
    r7 = 5.00000000000000000e-01;
    r6 = fmax(r6, r0);
    r8 = r6 * r6;
    r9 = -1.00000000000000000e+00;
    r10 = 2.00000000000000000e+00;
    r11 = r10 * r6;
    r12 = fmax(r0, r1);
    r13 = sqrt(r12);
    r11 = fma(r13, r11, r9 * r8);
    r11 = r1 <= r8 ? r1 : r11;
    r13 = 2.50000000000000000e+00;
    r14 = 1.00000000000000000e+00;
    r15 = 1.0 / r8;
    r15 = fma(r15, r1, r14);
    r14 = log(r15);
    r14 = r14 * r8;
    r11 = r5 < r13 ? r14 : r11;
    r14 = 1.50000000000000000e+00;
    r15 = sqrt(r15);
    r15 = r9 + r15;
    r15 = r10 * r15;
    r15 = r15 * r8;
    r11 = r5 < r14 ? r15 : r11;
    r11 = r5 < r7 ? r1 : r11;
    r11 = fmax(r3, r11);
    r11 = r2 * r11;
    r12 = 1.0 / r12;
    r11 = r11 * r12;
    r11 = sqrt(r11);
    r11 = r1 <= r0 ? r4 : r11;
    r11 = r11 * r11;
    r11 = r1 * r11;
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r11);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void ScalePriorScore(double* scale,
                     unsigned int scale_num_alloc,
                     SharedIndex* scale_indices,
                     double* inv_std,
                     unsigned int inv_std_num_alloc,
                     double* loss,
                     unsigned int loss_num_alloc,
                     double* const out_rTr,
                     size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ScalePriorScoreKernel<<<n_blocks, 1024>>>(scale,
                                            scale_num_alloc,
                                            scale_indices,
                                            inv_std,
                                            inv_std_num_alloc,
                                            loss,
                                            loss_num_alloc,
                                            out_rTr,
                                            problem_size);
}

}  // namespace caspar