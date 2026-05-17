#include "kernel_PinholeTranslation_start_w_contribute.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeTranslationStartWContributeKernel(
        double* PinholeTranslation_precond_diag,
        unsigned int PinholeTranslation_precond_diag_num_alloc,
        const double* const diag,
        double* PinholeTranslation_p,
        unsigned int PinholeTranslation_p_num_alloc,
        double* out_PinholeTranslation_w,
        unsigned int out_PinholeTranslation_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        PinholeTranslation_precond_diag,
        0 * PinholeTranslation_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1);
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r2;
    ReadIdx2<1024, double, double, double2>(PinholeTranslation_p,
                                            0 * PinholeTranslation_p_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r4);
    r0 = r0 * r3;
    r1 = r1 * r2;
    r1 = r1 * r4;
    AddIdx2<1024, double, double, double2>(
        out_PinholeTranslation_w,
        0 * out_PinholeTranslation_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx1<1024, double, double, double>(
        PinholeTranslation_precond_diag,
        2 * PinholeTranslation_precond_diag_num_alloc,
        global_thread_idx,
        r1);
    r2 = r1 * r2;
    ReadIdx1<1024, double, double, double>(PinholeTranslation_p,
                                           2 * PinholeTranslation_p_num_alloc,
                                           global_thread_idx,
                                           r1);
    r2 = r2 * r1;
    AddIdx1<1024, double, double, double>(
        out_PinholeTranslation_w,
        2 * out_PinholeTranslation_w_num_alloc,
        global_thread_idx,
        r2);
  };
}

void PinholeTranslationStartWContribute(
    double* PinholeTranslation_precond_diag,
    unsigned int PinholeTranslation_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeTranslation_p,
    unsigned int PinholeTranslation_p_num_alloc,
    double* out_PinholeTranslation_w,
    unsigned int out_PinholeTranslation_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeTranslationStartWContributeKernel<<<n_blocks, 1024>>>(
      PinholeTranslation_precond_diag,
      PinholeTranslation_precond_diag_num_alloc,
      diag,
      PinholeTranslation_p,
      PinholeTranslation_p_num_alloc,
      out_PinholeTranslation_w,
      out_PinholeTranslation_w_num_alloc,
      problem_size);
}

}  // namespace caspar