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
        double* PinholeTranslation_step,
        unsigned int PinholeTranslation_step_num_alloc,
        double* PinholeTranslation_precond_diag,
        unsigned int PinholeTranslation_precond_diag_num_alloc,
        const double* const diag,
        double* PinholeTranslation_njtr,
        unsigned int PinholeTranslation_njtr_num_alloc,
        double* const out_PinholeTranslation_pred_dec,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_PinholeTranslation_pred_dec_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        PinholeTranslation_step,
        0 * PinholeTranslation_step_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        PinholeTranslation_njtr,
        0 * PinholeTranslation_njtr_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        PinholeTranslation_precond_diag,
        0 * PinholeTranslation_precond_diag_num_alloc,
        global_thread_idx,
        r4,
        r5);
    r6 = r1 * r5;
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r6 = fma(r7, r6, r3);
    r3 = r0 * r4;
    r3 = fma(r7, r3, r2);
    r3 = fma(r0, r3, r1 * r6);
    ReadIdx1<1024, double, double, double>(
        PinholeTranslation_step,
        2 * PinholeTranslation_step_num_alloc,
        global_thread_idx,
        r6);
    ReadIdx1<1024, double, double, double>(
        PinholeTranslation_njtr,
        2 * PinholeTranslation_njtr_num_alloc,
        global_thread_idx,
        r2);
    ReadIdx1<1024, double, double, double>(
        PinholeTranslation_precond_diag,
        2 * PinholeTranslation_precond_diag_num_alloc,
        global_thread_idx,
        r8);
    r9 = r6 * r8;
    r9 = fma(r7, r9, r2);
    r3 = fma(r6, r9, r3);
  };
  SumStore<double>(out_PinholeTranslation_pred_dec_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r3);
  SumFlushFinal<double>(out_PinholeTranslation_pred_dec_local,
                        out_PinholeTranslation_pred_dec,
                        1);
}

void PinholeTranslationPredDecreaseTimesTwo(
    double* PinholeTranslation_step,
    unsigned int PinholeTranslation_step_num_alloc,
    double* PinholeTranslation_precond_diag,
    unsigned int PinholeTranslation_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeTranslation_njtr,
    unsigned int PinholeTranslation_njtr_num_alloc,
    double* const out_PinholeTranslation_pred_dec,
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