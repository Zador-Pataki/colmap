#include "kernel_pinhole_split_fixed_pose_fixed_focal_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedPoseFixedFocalResJacFirstKernel(
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* weight_loss,
        unsigned int weight_loss_num_alloc,
        float* pose,
        unsigned int pose_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r4,
                       r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r6, r7);
    r8 = -1.00000000000000000e+00;
    r6 = fmaf(r6, r8, r4);
    r4 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(
        pose, 4 * pose_num_alloc, global_thread_idx, r9, r10, r11);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(
        pose, 0 * pose_num_alloc, global_thread_idx, r15, r16, r17, r18);
    r19 = r16 * r17;
    r20 = 2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = r15 * r18;
    r21 = fmaf(r20, r21, r19);
    r11 = fmaf(r13, r21, r11);
    r22 = r15 * r17;
    r22 = r22 * r20;
    r23 = -2.00000000000000000e+00;
    r24 = r18 * r23;
    r25 = fmaf(r16, r24, r22);
    r26 = r15 * r15;
    r26 = r23 * r26;
    r27 = 1.00000000000000000e+00;
    r28 = r16 * r16;
    r28 = fmaf(r23, r28, r27);
    r29 = r26 + r28;
    r11 = fmaf(r12, r25, r11);
    r11 = fmaf(r14, r29, r11);
    r30 = copysign(1.0, r11);
    r30 = fmaf(r4, r30, r11);
    r11 = 1.0 / r30;
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r31, r32);
    r33 = r15 * r16;
    r33 = r33 * r20;
    r34 = fmaf(r17, r24, r33);
    r9 = fmaf(r13, r34, r9);
    r35 = r16 * r18;
    r35 = fmaf(r20, r35, r22);
    r22 = r17 * r17;
    r22 = r23 * r22;
    r28 = r22 + r28;
    r9 = fmaf(r14, r35, r9);
    r9 = fmaf(r12, r28, r9);
    r9 = r31 * r9;
    r6 = fmaf(r11, r9, r6);
    r7 = fmaf(r7, r8, r5);
    r5 = r17 * r18;
    r5 = fmaf(r20, r5, r33);
    r12 = fmaf(r12, r5, r10);
    r24 = fmaf(r15, r24, r19);
    r22 = r27 + r22;
    r22 = r22 + r26;
    r12 = fmaf(r14, r24, r12);
    r12 = fmaf(r13, r22, r12);
    r12 = r32 * r12;
    r7 = fmaf(r11, r12, r7);
    r13 = fmaf(r1, r7, r0 * r6);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r26,
                                         r19);
    r10 = 0.00000000000000000e+00;
    r19 = fmaxf(r19, r10);
    r33 = sqrtf(r19);
    r7 = fmaf(r3, r7, r2 * r6);
    r6 = fmaf(r13, r13, r7 * r7);
    r23 = 5.00000000000000000e-01;
    r26 = fmaxf(r26, r4);
    r36 = r26 * r26;
    r37 = r20 * r26;
    r38 = fmaxf(r4, r6);
    r39 = sqrtf(r38);
    r40 = r8 * r26;
    r40 = fmaf(r26, r40, r39 * r37);
    r40 = r6 <= r36 ? r6 : r40;
    r37 = 2.50000000000000000e+00;
    r39 = r26 * r26;
    r41 = 1.0 / r36;
    r41 = fmaf(r6, r41, r27);
    r42 = logf(r41);
    r39 = r39 * r42;
    r40 = r14 < r37 ? r39 : r40;
    r39 = 1.50000000000000000e+00;
    r42 = r20 * r26;
    r43 = sqrtf(r41);
    r43 = r8 + r43;
    r42 = r42 * r26;
    r42 = r42 * r43;
    r40 = r14 < r39 ? r42 : r40;
    r40 = r14 < r23 ? r6 : r40;
    r42 = fmaxf(r10, r40);
    r43 = 1.0 / r38;
    r43 = r19 * r43;
    r44 = r42 * r43;
    r45 = sqrtf(r44);
    r45 = r6 <= r4 ? r33 : r45;
    r33 = r13 * r45;
    r46 = r7 * r45;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r33, r46);
    r46 = r7 * r7;
    r46 = r46 * r45;
    r33 = r13 * r13;
    r33 = r33 * r45;
    r33 = fmaf(r45, r33, r45 * r46);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r33);
  if (global_thread_idx < problem_size) {
    r33 = r8 * r13;
    r46 = 2.50000000000000000e-01;
    r47 = r6 <= r36 ? r10 : r10;
    r47 = r14 < r37 ? r10 : r47;
    r47 = r14 < r39 ? r10 : r47;
    r47 = r14 < r23 ? r10 : r47;
    r47 = r46 * r47;
    r44 = rsqrtf(r44);
    r40 = copysign(1.0, r40);
    r40 = r27 + r40;
    r43 = r40 * r43;
    r47 = r47 * r44;
    r47 = r47 * r43;
    r47 = r6 <= r4 ? r10 : r47;
    r33 = r33 * r47;
    r40 = r2 * r20;
    r46 = r20 * r13;
    r40 = fmaf(r0, r46, r7 * r40);
    r48 = r23 * r26;
    r49 = -9.99999999999999955e-07;
    r49 = r49 + r6;
    r49 = copysign(1.0, r49);
    r49 = r27 + r49;
    r27 = rsqrtf(r38);
    r48 = r48 * r49;
    r48 = r48 * r27;
    r27 = r40 * r48;
    r27 = r6 <= r36 ? r40 : r27;
    r50 = 1.0 / r41;
    r51 = r40 * r50;
    r27 = r14 < r37 ? r51 : r27;
    r41 = rsqrtf(r41);
    r51 = r40 * r41;
    r27 = r14 < r39 ? r51 : r27;
    r27 = r14 < r23 ? r40 : r27;
    r51 = r23 * r27;
    r42 = r19 * r42;
    r19 = -5.00000000000000000e-01;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r42 = r42 * r19;
    r42 = r42 * r49;
    r42 = r42 * r38;
    r40 = fmaf(r40, r42, r43 * r51);
    r40 = r23 * r40;
    r40 = r40 * r44;
    r40 = r6 <= r4 ? r10 : r40;
    r51 = fmaf(r13, r40, r33);
    r51 = fmaf(r0, r45, r51);
    r38 = r8 * r7;
    r38 = r38 * r47;
    r40 = fmaf(r7, r40, r38);
    r40 = fmaf(r2, r45, r40);
    r47 = r3 * r20;
    r47 = fmaf(r1, r46, r7 * r47);
    r49 = r47 * r48;
    r49 = r6 <= r36 ? r47 : r49;
    r19 = r47 * r50;
    r49 = r14 < r37 ? r19 : r49;
    r19 = r47 * r41;
    r49 = r14 < r39 ? r19 : r49;
    r49 = r14 < r23 ? r47 : r49;
    r19 = r23 * r49;
    r47 = fmaf(r47, r42, r43 * r19);
    r47 = r23 * r47;
    r47 = r47 * r44;
    r47 = r6 <= r4 ? r10 : r47;
    r19 = fmaf(r13, r47, r33);
    r19 = fmaf(r1, r45, r19);
    r47 = fmaf(r7, r47, r38);
    r47 = fmaf(r3, r45, r47);
    WriteIdx4<1024, float, float, float4>(out_principal_point_jac,
                                          0 * out_principal_point_jac_num_alloc,
                                          global_thread_idx,
                                          r51,
                                          r40,
                                          r19,
                                          r47);
    r52 = r8 * r45;
    r53 = r7 * r52;
    r54 = r13 * r51;
    r54 = fmaf(r52, r54, r40 * r53);
    r55 = r13 * r19;
    r55 = fmaf(r47, r53, r52 * r55);
    WriteSum2<float, float>((float*)inout_shared, r54, r55);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r40, r40, r51 * r51);
    r54 = fmaf(r47, r47, r19 * r19);
    WriteSum2<float, float>((float*)inout_shared, r55, r54);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r51, r19, r40 * r47);
    WriteSum1<float, float>((float*)inout_shared, r47);
  };
  FlushSumShared<1, float>(out_principal_point_precond_tril,
                           0 * out_principal_point_precond_tril_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = r20 * r7;
    r40 = r31 * r28;
    r54 = r25 * r9;
    r30 = r30 * r30;
    r30 = 1.0 / r30;
    r30 = r8 * r30;
    r54 = fmaf(r30, r54, r11 * r40);
    r12 = r12 * r30;
    r40 = r32 * r5;
    r40 = fmaf(r11, r40, r25 * r12);
    r55 = fmaf(r3, r40, r2 * r54);
    r40 = fmaf(r1, r40, r0 * r54);
    r47 = fmaf(r40, r46, r55 * r47);
    r54 = r47 * r48;
    r54 = r6 <= r36 ? r47 : r54;
    r56 = r47 * r50;
    r54 = r14 < r37 ? r56 : r54;
    r56 = r47 * r41;
    r54 = r14 < r39 ? r56 : r54;
    r54 = r14 < r23 ? r47 : r54;
    r56 = r23 * r54;
    r56 = fmaf(r43, r56, r47 * r42);
    r56 = r23 * r56;
    r56 = r56 * r44;
    r56 = r6 <= r4 ? r10 : r56;
    r47 = fmaf(r13, r56, r33);
    r47 = fmaf(r45, r40, r47);
    r56 = fmaf(r7, r56, r38);
    r56 = fmaf(r45, r55, r56);
    r55 = r20 * r7;
    r40 = r21 * r9;
    r57 = r31 * r34;
    r57 = fmaf(r11, r57, r30 * r40);
    r40 = r32 * r22;
    r40 = fmaf(r21, r12, r11 * r40);
    r58 = fmaf(r3, r40, r2 * r57);
    r40 = fmaf(r1, r40, r0 * r57);
    r55 = fmaf(r40, r46, r58 * r55);
    r57 = r55 * r48;
    r57 = r6 <= r36 ? r55 : r57;
    r59 = r55 * r50;
    r57 = r14 < r37 ? r59 : r57;
    r59 = r55 * r41;
    r57 = r14 < r39 ? r59 : r57;
    r57 = r14 < r23 ? r55 : r57;
    r59 = r23 * r57;
    r55 = fmaf(r55, r42, r43 * r59);
    r55 = r23 * r55;
    r55 = r55 * r44;
    r55 = r6 <= r4 ? r10 : r55;
    r59 = fmaf(r13, r55, r33);
    r59 = fmaf(r45, r40, r59);
    r55 = fmaf(r7, r55, r38);
    r55 = fmaf(r45, r58, r55);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r47,
                                          r56,
                                          r59,
                                          r55);
    r58 = r20 * r7;
    r40 = r29 * r9;
    r60 = r31 * r35;
    r60 = fmaf(r11, r60, r30 * r40);
    r40 = r32 * r24;
    r40 = fmaf(r11, r40, r29 * r12);
    r12 = fmaf(r3, r40, r2 * r60);
    r40 = fmaf(r1, r40, r0 * r60);
    r46 = fmaf(r40, r46, r12 * r58);
    r48 = r46 * r48;
    r48 = r6 <= r36 ? r46 : r48;
    r50 = r46 * r50;
    r48 = r14 < r37 ? r50 : r48;
    r41 = r46 * r41;
    r48 = r14 < r39 ? r41 : r48;
    r48 = r14 < r23 ? r46 : r48;
    r14 = r23 * r48;
    r14 = fmaf(r43, r14, r46 * r42);
    r14 = r23 * r14;
    r14 = r14 * r44;
    r14 = r6 <= r4 ? r10 : r14;
    r33 = fmaf(r13, r14, r33);
    r33 = fmaf(r45, r40, r33);
    r14 = fmaf(r7, r14, r38);
    r14 = fmaf(r45, r12, r14);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r33,
                                          r14);
    r12 = r13 * r47;
    r12 = fmaf(r52, r12, r56 * r53);
    r45 = r13 * r59;
    r45 = fmaf(r55, r53, r52 * r45);
    r38 = r13 * r33;
    r53 = fmaf(r14, r53, r52 * r38);
    WriteSum3<float, float>((float*)inout_shared, r12, r45, r53);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fmaf(r56, r56, r47 * r47);
    r45 = fmaf(r55, r55, r59 * r59);
    r12 = fmaf(r33, r33, r14 * r14);
    WriteSum3<float, float>((float*)inout_shared, r53, r45, r12);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fmaf(r56, r55, r47 * r59);
    r56 = fmaf(r47, r33, r56 * r14);
    r14 = fmaf(r59, r33, r55 * r14);
    WriteSum3<float, float>((float*)inout_shared, r12, r56, r14);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedPoseFixedFocalResJacFirst(
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* weight_loss,
    unsigned int weight_loss_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedPoseFixedFocalResJacFirstKernel<<<n_blocks, 1024>>>(
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      weight_loss,
      weight_loss_num_alloc,
      pose,
      pose_num_alloc,
      focal,
      focal_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar