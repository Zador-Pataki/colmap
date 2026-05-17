#include "kernel_pinhole_split_fixed_rotation_fixed_focal_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedFocalFixedPointJtjnjtrDirectKernel(
        double* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        double* translation_jac,
        unsigned int translation_jac_num_alloc,
        double* principal_point_njtr,
        unsigned int principal_point_njtr_num_alloc,
        SharedIndex* principal_point_njtr_indices,
        double* principal_point_jac,
        unsigned int principal_point_jac_num_alloc,
        double* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_njtr_indices_loc[1024];
  translation_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex principal_point_njtr_indices_loc[1024];
  principal_point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            0 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
  };
  LoadShared<2, double, double>(principal_point_njtr,
                                0 * principal_point_njtr_num_alloc,
                                principal_point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        principal_point_njtr_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(principal_point_jac,
                                            0 * principal_point_jac_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    ReadIdx2<1024, double, double, double2>(principal_point_jac,
                                            2 * principal_point_jac_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
    r8 = fma(r3, r7, r2 * r5);
    r3 = fma(r3, r6, r2 * r4);
    r2 = fma(r0, r3, r1 * r8);
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            2 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r9,
                                            r10);
    r11 = fma(r9, r3, r10 * r8);
    WriteSum2<double, double>((double*)inout_shared, r2, r11);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            4 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r2);
    r3 = fma(r11, r3, r2 * r8);
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
    r9 = fma(r8, r9, r3 * r0);
  };
  LoadShared<1, double, double>(translation_njtr,
                                2 * translation_njtr_num_alloc,
                                translation_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared,
                        translation_njtr_indices_loc[threadIdx.x].target,
                        r0);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = fma(r0, r11, r9);
    r10 = fma(r8, r10, r3 * r1);
    r10 = fma(r0, r2, r10);
    r5 = fma(r5, r10, r4 * r9);
    r9 = fma(r6, r9, r7 * r10);
    WriteSum2<double, double>((double*)inout_shared, r5, r9);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedRotationFixedFocalFixedPointJtjnjtrDirect(
    double* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    double* translation_jac,
    unsigned int translation_jac_num_alloc,
    double* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    double* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedFocalFixedPointJtjnjtrDirectKernel<<<n_blocks,
                                                                     1024>>>(
      translation_njtr,
      translation_njtr_num_alloc,
      translation_njtr_indices,
      translation_jac,
      translation_jac_num_alloc,
      principal_point_njtr,
      principal_point_njtr_num_alloc,
      principal_point_njtr_indices,
      principal_point_jac,
      principal_point_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar