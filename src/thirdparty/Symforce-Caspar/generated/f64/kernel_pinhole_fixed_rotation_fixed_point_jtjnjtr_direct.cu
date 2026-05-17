#include "kernel_pinhole_fixed_rotation_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedRotationFixedPointJtjnjtrDirectKernel(
        double* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        double* translation_jac,
        unsigned int translation_jac_num_alloc,
        double* calib_njtr,
        unsigned int calib_njtr_num_alloc,
        SharedIndex* calib_njtr_indices,
        double* calib_jac,
        unsigned int calib_jac_num_alloc,
        double* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_njtr_indices_loc[1024];
  translation_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            0 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
  };
  LoadShared<2, double, double>(calib_njtr,
                                2 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 6 * calib_jac_num_alloc, global_thread_idx, r4, r5);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 4 * calib_jac_num_alloc, global_thread_idx, r6, r7);
    r8 = fma(r2, r7, r3 * r5);
  };
  LoadShared<2, double, double>(calib_njtr,
                                0 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r9,
                        r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r11, r12);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 2 * calib_jac_num_alloc, global_thread_idx, r13, r14);
    r8 = fma(r9, r12, r8);
    r8 = fma(r10, r14, r8);
    r3 = fma(r3, r4, r2 * r6);
    r3 = fma(r9, r11, r3);
    r3 = fma(r10, r13, r3);
    r10 = fma(r0, r3, r1 * r8);
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            2 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r2);
    r15 = fma(r9, r3, r2 * r8);
    WriteSum2<double, double>((double*)inout_shared, r10, r15);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            4 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r10);
    r3 = fma(r15, r3, r10 * r8);
    WriteSum1<double, double>((double*)inout_shared, r3);
  };
  FlushSumShared<1, double>(out_translation_njtr,
                            2 * out_translation_njtr_num_alloc,
                            translation_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(translation_njtr,
                                0 * translation_njtr_num_alloc,
                                translation_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        translation_njtr_indices_loc[threadIdx.x].target,
                        r3,
                        r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r8, r2, r3 * r1);
  };
  LoadShared<1, double, double>(translation_njtr,
                                2 * translation_njtr_num_alloc,
                                translation_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        translation_njtr_indices_loc[threadIdx.x].target,
                        r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r1, r10, r2);
    r9 = fma(r8, r9, r3 * r0);
    r9 = fma(r1, r15, r9);
    r11 = fma(r11, r9, r12 * r2);
    r14 = fma(r14, r2, r13 * r9);
    WriteSum2<double, double>((double*)inout_shared, r11, r14);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fma(r6, r9, r7 * r2);
    r9 = fma(r4, r9, r5 * r2);
    WriteSum2<double, double>((double*)inout_shared, r6, r9);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeFixedRotationFixedPointJtjnjtrDirect(
    double* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    double* translation_jac,
    unsigned int translation_jac_num_alloc,
    double* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    double* calib_jac,
    unsigned int calib_jac_num_alloc,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedRotationFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      translation_njtr,
      translation_njtr_num_alloc,
      translation_njtr_indices,
      translation_jac,
      translation_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar