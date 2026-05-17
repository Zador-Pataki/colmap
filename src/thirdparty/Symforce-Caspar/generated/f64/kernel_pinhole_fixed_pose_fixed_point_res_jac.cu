#include "kernel_pinhole_fixed_pose_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeFixedPoseFixedPointResJacKernel(
        double* calib,
        unsigned int calib_num_alloc,
        SharedIndex* calib_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_calib_precond_diag,
        unsigned int out_calib_precond_diag_num_alloc,
        double* const out_calib_precond_tril,
        unsigned int out_calib_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 0 * weight_loss_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(
      calib, 2 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fma(r4, r6, r2);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r2, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose, 4 * pose_num_alloc, global_thread_idx, r8, r9);
    ReadIdx2<1024, double, double, double2>(
        point, 0 * point_num_alloc, global_thread_idx, r10, r11);
    ReadIdx2<1024, double, double, double2>(
        pose, 0 * pose_num_alloc, global_thread_idx, r12, r13);
    r14 = r12 * r13;
    r15 = 2.00000000000000000e+00;
    r14 = r14 * r15;
    ReadIdx2<1024, double, double, double2>(
        pose, 2 * pose_num_alloc, global_thread_idx, r16, r17);
    r18 = -2.00000000000000000e+00;
    r19 = r17 * r18;
    r20 = fma(r16, r19, r14);
    r20 = fma(r11, r20, r8);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r8);
    r21 = r12 * r16;
    r21 = r21 * r15;
    r22 = r13 * r17;
    r22 = fma(r15, r22, r21);
    r23 = r16 * r16;
    r23 = r18 * r23;
    r24 = 1.00000000000000000e+00;
    r25 = r13 * r13;
    r25 = fma(r18, r25, r24);
    r26 = r23 + r25;
    r20 = fma(r8, r22, r20);
    r20 = fma(r10, r26, r20);
    r26 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r22);
    r27 = r13 * r16;
    r27 = r27 * r15;
    r28 = r12 * r17;
    r28 = fma(r15, r28, r27);
    r28 = fma(r11, r28, r22);
    r21 = fma(r13, r19, r21);
    r22 = r12 * r12;
    r22 = r18 * r22;
    r25 = r22 + r25;
    r28 = fma(r10, r21, r28);
    r28 = fma(r8, r25, r28);
    r25 = copysign(1.0, r28);
    r25 = fma(r26, r25, r28);
    r25 = 1.0 / r25;
    r20 = r20 * r25;
    r4 = fma(r2, r20, r4);
    r5 = fma(r5, r6, r3);
    r3 = r16 * r17;
    r3 = fma(r15, r3, r14);
    r3 = fma(r10, r3, r9);
    r19 = fma(r12, r19, r27);
    r23 = r24 + r23;
    r23 = r23 + r22;
    r3 = fma(r8, r19, r3);
    r3 = fma(r11, r23, r3);
    r23 = r7 * r3;
    r5 = fma(r25, r23, r5);
    r23 = fma(r1, r5, r0 * r4);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r11);
    r19 = 0.00000000000000000e+00;
    r11 = fmax(r11, r19);
    r8 = sqrt(r11);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r22, r27);
    r5 = fma(r27, r5, r22 * r4);
    r4 = fma(r23, r23, r5 * r5);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r10, r9);
    r14 = 5.00000000000000000e-01;
    r9 = fmax(r9, r26);
    r2 = r9 * r9;
    r28 = r15 * r9;
    r21 = fmax(r26, r4);
    r18 = sqrt(r21);
    r29 = r6 * r9;
    r29 = fma(r9, r29, r18 * r28);
    r29 = r4 <= r2 ? r4 : r29;
    r28 = 2.50000000000000000e+00;
    r18 = r9 * r9;
    r30 = 1.0 / r2;
    r30 = fma(r4, r30, r24);
    r31 = log(r30);
    r18 = r18 * r31;
    r29 = r10 < r28 ? r18 : r29;
    r18 = 1.50000000000000000e+00;
    r31 = r15 * r9;
    r32 = sqrt(r30);
    r32 = r6 + r32;
    r31 = r31 * r9;
    r31 = r31 * r32;
    r29 = r10 < r18 ? r31 : r29;
    r29 = r10 < r14 ? r4 : r29;
    r31 = fmax(r19, r29);
    r32 = 1.0 / r21;
    r32 = r11 * r32;
    r33 = r31 * r32;
    r34 = sqrt(r33);
    r34 = r4 <= r26 ? r8 : r34;
    r8 = r23 * r34;
    r35 = r5 * r34;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r8, r35);
    r35 = r6 * r23;
    r8 = 2.50000000000000000e-01;
    r36 = r4 <= r2 ? r19 : r19;
    r36 = r10 < r28 ? r19 : r36;
    r36 = r10 < r18 ? r19 : r36;
    r36 = r10 < r14 ? r19 : r36;
    r36 = r8 * r36;
    r33 = rsqrt(r33);
    r29 = copysign(1.0, r29);
    r29 = r24 + r29;
    r32 = r29 * r32;
    r36 = r36 * r33;
    r36 = r36 * r32;
    r36 = r4 <= r26 ? r19 : r36;
    r35 = r35 * r36;
    r29 = r15 * r23;
    r8 = r0 * r20;
    r37 = r22 * r15;
    r37 = r37 * r5;
    r37 = fma(r20, r37, r29 * r8);
    r38 = r14 * r9;
    r39 = -1.00000000000000008e-15;
    r39 = r39 + r4;
    r39 = copysign(1.0, r39);
    r39 = r24 + r39;
    r24 = rsqrt(r21);
    r38 = r38 * r39;
    r38 = r38 * r24;
    r24 = r37 * r38;
    r24 = r4 <= r2 ? r37 : r24;
    r40 = 1.0 / r30;
    r41 = r37 * r40;
    r24 = r10 < r28 ? r41 : r24;
    r30 = rsqrt(r30);
    r41 = r37 * r30;
    r24 = r10 < r18 ? r41 : r24;
    r24 = r10 < r14 ? r37 : r24;
    r41 = r14 * r24;
    r31 = r11 * r31;
    r11 = -5.00000000000000000e-01;
    r21 = r21 * r21;
    r21 = 1.0 / r21;
    r31 = r31 * r11;
    r31 = r31 * r39;
    r31 = r31 * r21;
    r37 = fma(r37, r31, r32 * r41);
    r37 = r14 * r37;
    r37 = r37 * r33;
    r37 = r4 <= r26 ? r19 : r37;
    r41 = fma(r23, r37, r35);
    r41 = fma(r34, r8, r41);
    r8 = r23 * r41;
    r21 = r6 * r34;
    r39 = r5 * r21;
    r11 = r6 * r5;
    r11 = r11 * r36;
    r37 = fma(r5, r37, r11);
    r36 = r22 * r34;
    r37 = fma(r20, r36, r37);
    r8 = fma(r37, r39, r21 * r8);
    r36 = r1 * r3;
    r36 = r36 * r25;
    r20 = r27 * r15;
    r20 = r20 * r3;
    r20 = r20 * r5;
    r20 = fma(r25, r20, r29 * r36);
    r42 = r20 * r38;
    r42 = r4 <= r2 ? r20 : r42;
    r43 = r20 * r40;
    r42 = r10 < r28 ? r43 : r42;
    r43 = r20 * r30;
    r42 = r10 < r18 ? r43 : r42;
    r42 = r10 < r14 ? r20 : r42;
    r43 = r14 * r42;
    r43 = fma(r32, r43, r20 * r31);
    r43 = r14 * r43;
    r43 = r43 * r33;
    r43 = r4 <= r26 ? r19 : r43;
    r20 = fma(r23, r43, r35);
    r20 = fma(r34, r36, r20);
    r36 = r23 * r20;
    r43 = fma(r5, r43, r11);
    r44 = r27 * r3;
    r44 = r44 * r34;
    r43 = fma(r25, r44, r43);
    r36 = fma(r43, r39, r21 * r36);
    WriteSum2<double, double>((double*)inout_shared, r8, r36);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r22, r34, r11);
    r8 = r22 * r15;
    r8 = fma(r0, r29, r5 * r8);
    r44 = r8 * r38;
    r44 = r4 <= r2 ? r8 : r44;
    r25 = r8 * r40;
    r44 = r10 < r28 ? r25 : r44;
    r25 = r8 * r30;
    r44 = r10 < r18 ? r25 : r44;
    r44 = r10 < r14 ? r8 : r44;
    r25 = r14 * r44;
    r8 = fma(r8, r31, r32 * r25);
    r8 = r14 * r8;
    r8 = r8 * r33;
    r8 = r4 <= r26 ? r19 : r8;
    r36 = fma(r5, r8, r36);
    r0 = fma(r0, r34, r35);
    r0 = fma(r23, r8, r0);
    r8 = r23 * r0;
    r8 = fma(r21, r8, r36 * r39);
    r25 = r27 * r15;
    r29 = fma(r1, r29, r5 * r25);
    r38 = r29 * r38;
    r38 = r4 <= r2 ? r29 : r38;
    r40 = r29 * r40;
    r38 = r10 < r28 ? r40 : r38;
    r30 = r29 * r30;
    r38 = r10 < r18 ? r30 : r38;
    r38 = r10 < r14 ? r29 : r38;
    r10 = r14 * r38;
    r10 = fma(r32, r10, r29 * r31);
    r10 = r14 * r10;
    r10 = r10 * r33;
    r10 = r4 <= r26 ? r19 : r10;
    r35 = fma(r23, r10, r35);
    r35 = fma(r1, r34, r35);
    r1 = r23 * r35;
    r10 = fma(r5, r10, r11);
    r10 = fma(r27, r34, r10);
    r39 = fma(r10, r39, r21 * r1);
    WriteSum2<double, double>((double*)inout_shared, r8, r39);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r41, r41, r37 * r37);
    r8 = fma(r20, r20, r43 * r43);
    WriteSum2<double, double>((double*)inout_shared, r39, r8);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fma(r0, r0, r36 * r36);
    r39 = fma(r10, r10, r35 * r35);
    WriteSum2<double, double>((double*)inout_shared, r8, r39);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fma(r37, r43, r41 * r20);
    r8 = fma(r37, r36, r41 * r0);
    WriteSum2<double, double>((double*)inout_shared, r39, r8);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r41, r35, r37 * r10);
    r8 = fma(r20, r0, r43 * r36);
    WriteSum2<double, double>((double*)inout_shared, r37, r8);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fma(r20, r35, r43 * r10);
    r10 = fma(r0, r35, r36 * r10);
    WriteSum2<double, double>((double*)inout_shared, r43, r10);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
}

void PinholeFixedPoseFixedPointResJac(
    double* calib,
    unsigned int calib_num_alloc,
    SharedIndex* calib_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    double* const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeFixedPoseFixedPointResJacKernel<<<n_blocks, 1024>>>(
      calib,
      calib_num_alloc,
      calib_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_calib_precond_diag,
      out_calib_precond_diag_num_alloc,
      out_calib_precond_tril,
      out_calib_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar