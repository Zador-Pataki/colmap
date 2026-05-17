#include "kernel_pinhole_split_fixed_pose_fixed_focal_fixed_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalFixedPointResJacKernel(
        double* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        double* pixel,
        unsigned int pixel_num_alloc,
        double* weight_loss,
        unsigned int weight_loss_num_alloc,
        double* pose,
        unsigned int pose_num_alloc,
        double* focal,
        unsigned int focal_num_alloc,
        double* point,
        unsigned int point_num_alloc,
        double* out_res,
        unsigned int out_res_num_alloc,
        double* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        double* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        double* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 0 * weight_loss_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(principal_point,
                                0 * principal_point_num_alloc,
                                principal_point_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        principal_point_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r5 = fma(r5, r6, r3);
    ReadIdx2<1024, double, double, double2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r3, r7);
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
    r18 = r16 * r15;
    r19 = fma(r17, r18, r14);
    r19 = fma(r10, r19, r9);
    ReadIdx1<1024, double, double, double>(
        point, 2 * point_num_alloc, global_thread_idx, r9);
    r20 = r13 * r18;
    r21 = -2.00000000000000000e+00;
    r22 = r17 * r21;
    r23 = fma(r12, r22, r20);
    r24 = r16 * r16;
    r24 = r24 * r21;
    r25 = 1.00000000000000000e+00;
    r26 = r12 * r12;
    r26 = fma(r21, r26, r25);
    r27 = r24 + r26;
    r19 = fma(r9, r23, r19);
    r19 = fma(r11, r27, r19);
    r27 = r7 * r19;
    r23 = 1.00000000000000008e-15;
    ReadIdx1<1024, double, double, double>(
        pose, 6 * pose_num_alloc, global_thread_idx, r28);
    r29 = r12 * r17;
    r29 = fma(r15, r29, r20);
    r29 = fma(r11, r29, r28);
    r18 = r12 * r18;
    r28 = fma(r13, r22, r18);
    r20 = r13 * r13;
    r20 = r21 * r20;
    r26 = r20 + r26;
    r29 = fma(r10, r28, r29);
    r29 = fma(r9, r26, r29);
    r26 = copysign(1.0, r29);
    r26 = fma(r23, r26, r29);
    r26 = 1.0 / r26;
    r5 = fma(r26, r27, r5);
    r4 = fma(r4, r6, r2);
    r22 = fma(r16, r22, r14);
    r22 = fma(r11, r22, r8);
    r11 = r13 * r17;
    r11 = fma(r15, r11, r18);
    r24 = r25 + r24;
    r24 = r24 + r20;
    r22 = fma(r9, r11, r22);
    r22 = fma(r10, r24, r22);
    r24 = r3 * r22;
    r4 = fma(r26, r24, r4);
    r24 = fma(r0, r4, r1 * r5);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r26);
    r10 = 0.00000000000000000e+00;
    r26 = fmax(r26, r10);
    r11 = sqrt(r26);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r9, r20);
    r4 = fma(r9, r4, r20 * r5);
    r5 = fma(r4, r4, r24 * r24);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r18, r8);
    r16 = 5.00000000000000000e-01;
    r8 = fmax(r8, r23);
    r14 = r8 * r8;
    r2 = r15 * r8;
    r27 = fmax(r23, r5);
    r29 = sqrt(r27);
    r2 = fma(r29, r2, r6 * r14);
    r2 = r5 <= r14 ? r5 : r2;
    r29 = 2.50000000000000000e+00;
    r28 = 1.0 / r14;
    r28 = fma(r5, r28, r25);
    r21 = log(r28);
    r21 = r21 * r14;
    r2 = r18 < r29 ? r21 : r2;
    r21 = 1.50000000000000000e+00;
    r30 = sqrt(r28);
    r30 = r6 + r30;
    r30 = r15 * r30;
    r30 = r30 * r14;
    r2 = r18 < r21 ? r30 : r2;
    r2 = r18 < r16 ? r5 : r2;
    r30 = fmax(r10, r2);
    r31 = 1.0 / r27;
    r31 = r26 * r31;
    r32 = r30 * r31;
    r33 = sqrt(r32);
    r33 = r5 <= r23 ? r11 : r33;
    r11 = r24 * r33;
    r34 = r4 * r33;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r11, r34);
    r34 = r6 * r24;
    r11 = 2.50000000000000000e-01;
    r35 = r5 <= r14 ? r10 : r10;
    r35 = r18 < r29 ? r10 : r35;
    r35 = r18 < r21 ? r10 : r35;
    r35 = r18 < r16 ? r10 : r35;
    r35 = r11 * r35;
    r32 = rsqrt(r32);
    r2 = copysign(1.0, r2);
    r2 = r25 + r2;
    r31 = r2 * r31;
    r35 = r35 * r32;
    r35 = r35 * r31;
    r35 = r5 <= r23 ? r10 : r35;
    r34 = r34 * r35;
    r2 = r0 * r15;
    r11 = r9 * r15;
    r11 = fma(r4, r11, r24 * r2);
    r30 = r26 * r30;
    r26 = -5.00000000000000000e-01;
    r2 = -1.00000000000000008e-15;
    r2 = r2 + r5;
    r2 = copysign(1.0, r2);
    r2 = r25 + r2;
    r25 = r27 * r27;
    r25 = 1.0 / r25;
    r30 = r30 * r26;
    r30 = r30 * r2;
    r30 = r30 * r25;
    r25 = r16 * r8;
    r27 = rsqrt(r27);
    r25 = r25 * r2;
    r25 = r25 * r27;
    r27 = r11 * r25;
    r27 = r5 <= r14 ? r11 : r27;
    r2 = 1.0 / r28;
    r26 = r11 * r2;
    r27 = r18 < r29 ? r26 : r27;
    r28 = rsqrt(r28);
    r26 = r11 * r28;
    r27 = r18 < r21 ? r26 : r27;
    r27 = r18 < r16 ? r11 : r27;
    r26 = r16 * r27;
    r26 = fma(r31, r26, r11 * r30);
    r26 = r16 * r26;
    r26 = r26 * r32;
    r26 = r5 <= r23 ? r10 : r26;
    r11 = fma(r24, r26, r34);
    r11 = fma(r0, r33, r11);
    r36 = r24 * r11;
    r37 = r6 * r33;
    r6 = r6 * r4;
    r6 = r6 * r35;
    r26 = fma(r4, r26, r6);
    r26 = fma(r9, r33, r26);
    r35 = r4 * r37;
    r36 = fma(r26, r35, r37 * r36);
    r38 = r1 * r15;
    r39 = r20 * r15;
    r39 = fma(r4, r39, r24 * r38);
    r25 = r39 * r25;
    r25 = r5 <= r14 ? r39 : r25;
    r2 = r39 * r2;
    r25 = r18 < r29 ? r2 : r25;
    r28 = r39 * r28;
    r25 = r18 < r21 ? r28 : r25;
    r25 = r18 < r16 ? r39 : r25;
    r18 = r16 * r25;
    r18 = fma(r31, r18, r39 * r30);
    r18 = r16 * r18;
    r18 = r18 * r32;
    r18 = r5 <= r23 ? r10 : r18;
    r4 = fma(r4, r18, r6);
    r4 = fma(r20, r33, r4);
    r18 = fma(r24, r18, r34);
    r18 = fma(r1, r33, r18);
    r33 = r24 * r18;
    r33 = fma(r37, r33, r4 * r35);
    WriteSum2<double, double>((double*)inout_shared, r36, r33);
  };
  FlushSumShared<2, double>(out_principal_point_njtr,
                            0 * out_principal_point_njtr_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fma(r11, r11, r26 * r26);
    r36 = fma(r18, r18, r4 * r4);
    WriteSum2<double, double>((double*)inout_shared, r33, r36);
  };
  FlushSumShared<2, double>(out_principal_point_precond_diag,
                            0 * out_principal_point_precond_diag_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = fma(r11, r18, r26 * r4);
    WriteSum1<double, double>((double*)inout_shared, r4);
  };
  FlushSumShared<1, double>(out_principal_point_precond_tril,
                            0 * out_principal_point_precond_tril_num_alloc,
                            principal_point_indices_loc,
                            (double*)inout_shared);
}

void PinholeSplitFixedPoseFixedFocalFixedPointResJac(
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* weight_loss,
    unsigned int weight_loss_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    double* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    double* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalFixedPointResJacKernel<<<n_blocks, 1024>>>(
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      point,
      point_num_alloc,
      out_res,
      out_res_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar