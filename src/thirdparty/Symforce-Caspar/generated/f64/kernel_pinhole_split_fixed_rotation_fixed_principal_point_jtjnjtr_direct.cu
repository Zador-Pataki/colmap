#include "kernel_pinhole_split_fixed_rotation_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedRotationFixedPrincipalPointJtjnjtrDirectKernel(
        double* translation_njtr,
        unsigned int translation_njtr_num_alloc,
        SharedIndex* translation_njtr_indices,
        double* translation_jac,
        unsigned int translation_jac_num_alloc,
        double* focal_njtr,
        unsigned int focal_njtr_num_alloc,
        SharedIndex* focal_njtr_indices,
        double* focal_jac,
        unsigned int focal_jac_num_alloc,
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_translation_njtr,
        unsigned int out_translation_njtr_num_alloc,
        double* const out_focal_njtr,
        unsigned int out_focal_njtr_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex translation_njtr_indices_loc[1024];
  translation_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? translation_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_njtr_indices_loc[1024];
  focal_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            0 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
  };
  LoadShared<1, double, double>(point_njtr,
                                2 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r3, r4);
  };
  LoadShared<2, double, double>(point_njtr,
                                0 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        point_njtr_indices_loc[threadIdx.x].target,
                        r5,
                        r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r7, r8);
    r9 = fma(r6, r7, r2 * r3);
    ReadIdx2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r10, r11);
    r9 = fma(r5, r10, r9);
  };
  LoadShared<2, double, double>(focal_njtr,
                                0 * focal_njtr_num_alloc,
                                focal_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_njtr_indices_loc[threadIdx.x].target,
                        r12,
                        r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        focal_jac, 2 * focal_jac_num_alloc, global_thread_idx, r14, r15);
    ReadIdx2<1024, double, double, double2>(
        focal_jac, 0 * focal_jac_num_alloc, global_thread_idx, r16, r17);
    r18 = fma(r12, r16, r13 * r14);
    r19 = r9 + r18;
    r6 = fma(r6, r8, r2 * r4);
    r6 = fma(r5, r11, r6);
    r13 = fma(r13, r15, r12 * r17);
    r12 = r6 + r13;
    r5 = fma(r1, r12, r0 * r19);
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            2 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r2,
                                            r20);
    r21 = fma(r20, r12, r2 * r19);
    WriteSum2<double, double>((double*)inout_shared, r5, r21);
  };
  FlushSumShared<2, double>(out_translation_njtr,
                            0 * out_translation_njtr_num_alloc,
                            translation_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(translation_jac,
                                            4 * translation_jac_num_alloc,
                                            global_thread_idx,
                                            r21,
                                            r5);
    r12 = fma(r5, r12, r21 * r19);
    WriteSum1<double, double>((double*)inout_shared, r12);
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
                        r12,
                        r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = fma(r19, r20, r12 * r1);
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
    r20 = fma(r1, r5, r20);
    r6 = r6 + r20;
    r2 = fma(r19, r2, r12 * r0);
    r2 = fma(r1, r21, r2);
    r9 = r9 + r2;
    r16 = fma(r16, r9, r17 * r6);
    r9 = fma(r14, r9, r15 * r6);
    WriteSum2<double, double>((double*)inout_shared, r16, r9);
  };
  FlushSumShared<2, double>(out_focal_njtr,
                            0 * out_focal_njtr_num_alloc,
                            focal_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r18 = r2 + r18;
    r13 = r20 + r13;
    r11 = fma(r11, r13, r10 * r18);
    r8 = fma(r8, r13, r7 * r18);
    WriteSum2<double, double>((double*)inout_shared, r11, r8);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fma(r4, r13, r3 * r18);
    WriteSum1<double, double>((double*)inout_shared, r13);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedRotationFixedPrincipalPointJtjnjtrDirect(
    double* translation_njtr,
    unsigned int translation_njtr_num_alloc,
    SharedIndex* translation_njtr_indices,
    double* translation_jac,
    unsigned int translation_jac_num_alloc,
    double* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    double* focal_jac,
    unsigned int focal_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_translation_njtr,
    unsigned int out_translation_njtr_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedRotationFixedPrincipalPointJtjnjtrDirectKernel<<<n_blocks,
                                                                    1024>>>(
      translation_njtr,
      translation_njtr_num_alloc,
      translation_njtr_indices,
      translation_jac,
      translation_jac_num_alloc,
      focal_njtr,
      focal_njtr_num_alloc,
      focal_njtr_indices,
      focal_jac,
      focal_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_translation_njtr,
      out_translation_njtr_num_alloc,
      out_focal_njtr,
      out_focal_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar