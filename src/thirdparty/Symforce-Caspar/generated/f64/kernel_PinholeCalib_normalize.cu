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
    PinholeCalibNormalizeKernel(double* precond_diag,
                                unsigned int precond_diag_num_alloc,
                                double* precond_tril,
                                unsigned int precond_tril_num_alloc,
                                double* njtr,
                                unsigned int njtr_num_alloc,
                                const double* const diag,
                                double* out_normalized,
                                unsigned int out_normalized_num_alloc,
                                size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20;
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r1 = 1.00000000000000008e-15;
    r1 = r0 * r1;
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 0 * precond_diag_num_alloc, global_thread_idx, r2, r3);
    r4 = 1.00000000000000000e+00;
    r4 = r0 + r4;
    r3 = fma(r3, r4, r1);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 0 * precond_tril_num_alloc, global_thread_idx, r0, r5);
    r6 = -1.00000000000000000e+00;
    r2 = fma(r2, r4, r1);
    r2 = 1.0 / r2;
    r7 = r6 * r2;
    r8 = r0 * r7;
    r3 = fma(r0, r8, r3);
    r3 = 1.0 / r3;
    ReadIdx2<1024, double, double, double2>(
        njtr, 0 * njtr_num_alloc, global_thread_idx, r0, r9);
    r9 = fma(r0, r8, r9);
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 2 * precond_tril_num_alloc, global_thread_idx, r10, r11);
    r11 = fma(r5, r8, r11);
    r12 = r11 * r3;
    r13 = r6 * r12;
    r14 = r5 * r5;
    r11 = fma(r11, r12, r2 * r14);
    r11 = fma(r6, r11, r1);
    ReadIdx2<1024, double, double, double2>(
        precond_diag, 2 * precond_diag_num_alloc, global_thread_idx, r14, r15);
    r11 = fma(r14, r4, r11);
    r11 = 1.0 / r11;
    ReadIdx2<1024, double, double, double2>(
        njtr, 2 * njtr_num_alloc, global_thread_idx, r14, r16);
    r14 = fma(r9, r13, r14);
    r17 = r5 * r0;
    r14 = fma(r7, r17, r14);
    r17 = r6 * r14;
    ReadIdx2<1024, double, double, double2>(
        precond_tril, 4 * precond_tril_num_alloc, global_thread_idx, r18, r19);
    r20 = r5 * r10;
    r18 = fma(r10, r8, r18);
    r12 = fma(r18, r12, r2 * r20);
    r12 = fma(r6, r12, r19);
    r19 = r12 * r11;
    r17 = fma(r19, r17, r16);
    r16 = r6 * r18;
    r16 = r16 * r9;
    r17 = fma(r3, r16, r17);
    r20 = r10 * r0;
    r17 = fma(r7, r20, r17);
    r20 = r10 * r10;
    r12 = fma(r12, r19, r2 * r20);
    r20 = r18 * r18;
    r12 = fma(r3, r20, r12);
    r12 = fma(r6, r12, r1);
    r12 = fma(r15, r4, r12);
    r12 = 1.0 / r12;
    r12 = r17 * r12;
    r17 = r6 * r12;
    r17 = fma(r19, r17, r14 * r11);
    r13 = fma(r17, r13, r9 * r3);
    r9 = r6 * r18;
    r9 = r9 * r3;
    r13 = fma(r12, r9, r13);
    r9 = r10 * r7;
    r9 = fma(r12, r9, r0 * r2);
    r2 = r5 * r17;
    r9 = fma(r7, r2, r9);
    r9 = fma(r13, r8, r9);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             0 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r9,
                                             r13);
    WriteIdx2<1024, double, double, double2>(out_normalized,
                                             2 * out_normalized_num_alloc,
                                             global_thread_idx,
                                             r17,
                                             r12);
  };
}

void PinholeCalibNormalize(double* precond_diag,
                           unsigned int precond_diag_num_alloc,
                           double* precond_tril,
                           unsigned int precond_tril_num_alloc,
                           double* njtr,
                           unsigned int njtr_num_alloc,
                           const double* const diag,
                           double* out_normalized,
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