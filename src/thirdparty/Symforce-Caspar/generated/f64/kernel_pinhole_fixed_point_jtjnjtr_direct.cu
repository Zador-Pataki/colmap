#include "kernel_pinhole_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPointJtjnjtrDirectKernel(double* pose_njtr,
                                         unsigned int pose_njtr_num_alloc,
                                         SharedIndex* pose_njtr_indices,
                                         double* pose_jac,
                                         unsigned int pose_jac_num_alloc,
                                         double* calib_njtr,
                                         unsigned int calib_njtr_num_alloc,
                                         SharedIndex* calib_njtr_indices,
                                         double* calib_jac,
                                         unsigned int calib_jac_num_alloc,
                                         double* const out_pose_njtr,
                                         unsigned int out_pose_njtr_num_alloc,
                                         double* const out_calib_njtr,
                                         unsigned int out_calib_njtr_num_alloc,
                                         size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
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
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r9, r2);
    r15 = fma(r9, r3, r2 * r8);
    WriteSum2<double, double>((double*)inout_shared, r10, r15);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r15, r10);
    r16 = fma(r15, r3, r10 * r8);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r17, r3, r18 * r8);
    WriteSum2<double, double>((double*)inout_shared, r16, r19);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r19, r16);
    r20 = fma(r19, r3, r16 * r8);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r21, r22);
    r8 = fma(r22, r8, r21 * r3);
    WriteSum2<double, double>((double*)inout_shared, r20, r8);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(pose_njtr,
                                4 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r8,
                        r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = fma(r8, r19, r20 * r21);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r21,
                        r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose_njtr,
                                0 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r23,
                        r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r19 = fma(r21, r15, r19);
    r19 = fma(r3, r17, r19);
    r19 = fma(r23, r0, r19);
    r19 = fma(r24, r9, r19);
    r18 = fma(r3, r18, r20 * r22);
    r18 = fma(r21, r10, r18);
    r18 = fma(r23, r1, r18);
    r18 = fma(r24, r2, r18);
    r18 = fma(r8, r16, r18);
    r12 = fma(r12, r18, r11 * r19);
    r14 = fma(r14, r18, r13 * r19);
    WriteSum2<double, double>((double*)inout_shared, r12, r14);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r7, r18, r6 * r19);
    r18 = fma(r5, r18, r4 * r19);
    WriteSum2<double, double>((double*)inout_shared, r7, r18);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
}

void PinholeFixedPointJtjnjtrDirect(double* pose_njtr,
                                    unsigned int pose_njtr_num_alloc,
                                    SharedIndex* pose_njtr_indices,
                                    double* pose_jac,
                                    unsigned int pose_jac_num_alloc,
                                    double* calib_njtr,
                                    unsigned int calib_njtr_num_alloc,
                                    SharedIndex* calib_njtr_indices,
                                    double* calib_jac,
                                    unsigned int calib_jac_num_alloc,
                                    double* const out_pose_njtr,
                                    unsigned int out_pose_njtr_num_alloc,
                                    double* const out_calib_njtr,
                                    unsigned int out_calib_njtr_num_alloc,
                                    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar