#include "kernel_PinholeTranslation_alpha_denominator_or_beta_numerator.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeTranslationAlphaDenominatorOrBetaNumeratorKernel(
        double* PinholeTranslation_p_kp1,
        unsigned int PinholeTranslation_p_kp1_num_alloc,
        double* PinholeTranslation_w,
        unsigned int PinholeTranslation_w_num_alloc,
        double* const PinholeTranslation_out,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[256];

  __shared__ double PinholeTranslation_out_local[1];

  double r0, r1, r2, r3, r4, r5;

  if (global_thread_idx < problem_size) {
    ReadIdx1<1024, double, double, double>(
        PinholeTranslation_p_kp1,
        2 * PinholeTranslation_p_kp1_num_alloc,
        global_thread_idx,
        r0);
    ReadIdx1<1024, double, double, double>(PinholeTranslation_w,
                                           2 * PinholeTranslation_w_num_alloc,
                                           global_thread_idx,
                                           r1);
    ReadIdx2<1024, double, double, double2>(
        PinholeTranslation_p_kp1,
        0 * PinholeTranslation_p_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(PinholeTranslation_w,
                                            0 * PinholeTranslation_w_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    r5 = fma(r3, r5, r0 * r1);
    r5 = fma(r2, r4, r5);
  };
  SumStore<double>(PinholeTranslation_out_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r5);
  SumFlushFinal<double>(
      PinholeTranslation_out_local, PinholeTranslation_out, 1);
}

void PinholeTranslationAlphaDenominatorOrBetaNumerator(
    double* PinholeTranslation_p_kp1,
    unsigned int PinholeTranslation_p_kp1_num_alloc,
    double* PinholeTranslation_w,
    unsigned int PinholeTranslation_w_num_alloc,
    double* const PinholeTranslation_out,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeTranslationAlphaDenominatorOrBetaNumeratorKernel<<<n_blocks, 1024>>>(
      PinholeTranslation_p_kp1,
      PinholeTranslation_p_kp1_num_alloc,
      PinholeTranslation_w,
      PinholeTranslation_w_num_alloc,
      PinholeTranslation_out,
      problem_size);
}

}  // namespace caspar