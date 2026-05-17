#include "kernel_pinhole_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeResJacKernel(float* pose,
                        unsigned int pose_num_alloc,
                        SharedIndex* pose_indices,
                        float* calib,
                        unsigned int calib_num_alloc,
                        SharedIndex* calib_indices,
                        float* point,
                        unsigned int point_num_alloc,
                        SharedIndex* point_indices,
                        float* pixel,
                        unsigned int pixel_num_alloc,
                        float* weight_loss,
                        unsigned int weight_loss_num_alloc,
                        float* out_res,
                        unsigned int out_res_num_alloc,
                        float* out_pose_jac,
                        unsigned int out_pose_jac_num_alloc,
                        float* const out_pose_njtr,
                        unsigned int out_pose_njtr_num_alloc,
                        float* const out_pose_precond_diag,
                        unsigned int out_pose_precond_diag_num_alloc,
                        float* const out_pose_precond_tril,
                        unsigned int out_pose_precond_tril_num_alloc,
                        float* out_calib_jac,
                        unsigned int out_calib_jac_num_alloc,
                        float* const out_calib_njtr,
                        unsigned int out_calib_njtr_num_alloc,
                        float* const out_calib_precond_diag,
                        unsigned int out_calib_precond_diag_num_alloc,
                        float* const out_calib_precond_tril,
                        unsigned int out_calib_precond_tril_num_alloc,
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

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79;

  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(weight_loss,
                                         0 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1,
                                         r2,
                                         r3);
  };
  LoadShared<4, float, float>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       calib_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r8, r9);
    r10 = -1.00000000000000000e+00;
    r8 = fmaf(r8, r10, r6);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r6,
                       r11,
                       r12);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r16,
                       r17,
                       r18,
                       r19);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r20 = r16 * r17;
    r21 = 2.00000000000000000e+00;
    r20 = r20 * r21;
    r22 = -2.00000000000000000e+00;
    r23 = r18 * r22;
    r24 = r19 * r23;
    r25 = r20 + r24;
    r6 = fmaf(r14, r25, r6);
    r26 = r16 * r18;
    r26 = r26 * r21;
    r27 = r17 * r19;
    r27 = r27 * r21;
    r28 = r26 + r27;
    r29 = r18 * r23;
    r30 = 1.00000000000000000e+00;
    r31 = r17 * r17;
    r32 = fmaf(r22, r31, r30);
    r33 = r29 + r32;
    r6 = fmaf(r15, r28, r6);
    r6 = fmaf(r13, r33, r6);
    r34 = 9.99999999999999955e-07;
    r35 = r17 * r18;
    r35 = r35 * r21;
    r36 = r16 * r19;
    r36 = r36 * r21;
    r37 = r35 + r36;
    r12 = fmaf(r14, r37, r12);
    r38 = r17 * r19;
    r38 = r38 * r22;
    r26 = r26 + r38;
    r39 = r16 * r16;
    r40 = r22 * r39;
    r32 = r40 + r32;
    r12 = fmaf(r13, r26, r12);
    r12 = fmaf(r15, r32, r12);
    r41 = copysign(1.0, r12);
    r41 = fmaf(r34, r41, r12);
    r12 = 1.0 / r41;
    r42 = r4 * r12;
    r8 = fmaf(r6, r42, r8);
    r9 = fmaf(r9, r10, r7);
    r7 = r18 * r19;
    r7 = r7 * r21;
    r20 = r20 + r7;
    r11 = fmaf(r13, r20, r11);
    r43 = r16 * r19;
    r43 = r43 * r22;
    r35 = r35 + r43;
    r29 = r30 + r29;
    r29 = r29 + r40;
    r11 = fmaf(r15, r35, r11);
    r11 = fmaf(r14, r29, r11);
    r40 = r5 * r11;
    r9 = fmaf(r12, r40, r9);
    r44 = fmaf(r1, r9, r0 * r8);
    ReadIdx3<1024, float, float, float4>(weight_loss,
                                         4 * weight_loss_num_alloc,
                                         global_thread_idx,
                                         r45,
                                         r46,
                                         r47);
    r48 = 0.00000000000000000e+00;
    r47 = fmaxf(r47, r48);
    r49 = sqrtf(r47);
    r9 = fmaf(r3, r9, r2 * r8);
    r8 = fmaf(r9, r9, r44 * r44);
    r50 = 5.00000000000000000e-01;
    r46 = fmaxf(r46, r34);
    r51 = r46 * r46;
    r52 = r21 * r46;
    r53 = fmaxf(r34, r8);
    r54 = sqrtf(r53);
    r55 = r10 * r46;
    r55 = fmaf(r46, r55, r54 * r52);
    r55 = r8 <= r51 ? r8 : r55;
    r52 = 2.50000000000000000e+00;
    r54 = r46 * r46;
    r56 = 1.0 / r51;
    r56 = fmaf(r8, r56, r30);
    r57 = logf(r56);
    r54 = r54 * r57;
    r55 = r45 < r52 ? r54 : r55;
    r54 = 1.50000000000000000e+00;
    r57 = r21 * r46;
    r58 = sqrtf(r56);
    r58 = r10 + r58;
    r57 = r57 * r46;
    r57 = r57 * r58;
    r55 = r45 < r54 ? r57 : r55;
    r55 = r45 < r50 ? r8 : r55;
    r57 = fmaxf(r48, r55);
    r58 = 1.0 / r53;
    r58 = r47 * r58;
    r59 = r57 * r58;
    r60 = sqrtf(r59);
    r60 = r8 <= r34 ? r49 : r60;
    r49 = r44 * r60;
    r61 = r9 * r60;
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r49, r61);
    r57 = r47 * r57;
    r47 = -5.00000000000000000e-01;
    r61 = -9.99999999999999955e-07;
    r61 = r61 + r8;
    r61 = copysign(1.0, r61);
    r61 = r30 + r61;
    r49 = r53 * r53;
    r49 = 1.0 / r49;
    r57 = r57 * r47;
    r57 = r57 * r61;
    r57 = r57 * r49;
    r49 = r21 * r9;
    r47 = r17 * r23;
    r43 = r43 + r47;
    r62 = r18 * r18;
    r63 = r10 * r31;
    r64 = r62 + r63;
    r19 = r19 * r19;
    r65 = r10 * r39;
    r66 = r19 + r65;
    r67 = r64 + r66;
    r67 = fmaf(r14, r67, r15 * r43);
    r43 = r10 * r67;
    r41 = r41 * r41;
    r41 = 1.0 / r41;
    r43 = r43 * r41;
    r68 = r10 * r19;
    r69 = r39 + r68;
    r64 = r64 + r69;
    r64 = fmaf(r15, r64, r14 * r35);
    r70 = r5 * r64;
    r70 = fmaf(r12, r70, r40 * r43);
    r17 = r16 * r17;
    r17 = r17 * r22;
    r7 = r7 + r17;
    r7 = fmaf(r14, r28, r15 * r7);
    r4 = r4 * r6;
    r4 = r4 * r10;
    r4 = r4 * r41;
    r7 = fmaf(r67, r4, r7 * r42);
    r22 = fmaf(r2, r7, r3 * r70);
    r7 = fmaf(r0, r7, r1 * r70);
    r70 = r21 * r44;
    r49 = fmaf(r7, r70, r22 * r49);
    r43 = r50 * r46;
    r53 = rsqrtf(r53);
    r43 = r43 * r61;
    r43 = r43 * r53;
    r53 = r49 * r43;
    r53 = r8 <= r51 ? r49 : r53;
    r61 = 1.0 / r56;
    r71 = r49 * r61;
    r53 = r45 < r52 ? r71 : r53;
    r56 = rsqrtf(r56);
    r71 = r49 * r56;
    r53 = r45 < r54 ? r71 : r53;
    r53 = r45 < r50 ? r49 : r53;
    r71 = r50 * r53;
    r55 = copysign(1.0, r55);
    r55 = r30 + r55;
    r58 = r55 * r58;
    r71 = fmaf(r58, r71, r49 * r57);
    r71 = r50 * r71;
    r59 = rsqrtf(r59);
    r71 = r71 * r59;
    r71 = r8 <= r34 ? r48 : r71;
    r7 = fmaf(r60, r7, r44 * r71);
    r49 = r10 * r44;
    r55 = 2.50000000000000000e-01;
    r30 = r8 <= r51 ? r48 : r48;
    r30 = r45 < r52 ? r48 : r30;
    r30 = r45 < r54 ? r48 : r30;
    r30 = r45 < r50 ? r48 : r30;
    r30 = r55 * r30;
    r30 = r30 * r59;
    r30 = r30 * r58;
    r30 = r8 <= r34 ? r48 : r30;
    r49 = r49 * r30;
    r7 = r7 + r49;
    r22 = fmaf(r60, r22, r9 * r71);
    r71 = r10 * r9;
    r71 = r71 * r30;
    r22 = r22 + r71;
    r23 = r16 * r23;
    r38 = r38 + r23;
    r19 = r39 + r19;
    r18 = r18 * r18;
    r18 = r18 * r10;
    r19 = r19 + r63;
    r19 = r19 + r18;
    r19 = fmaf(r15, r19, r13 * r38);
    r18 = r31 + r18;
    r69 = r69 + r18;
    r69 = fmaf(r13, r69, r15 * r26);
    r19 = fmaf(r69, r4, r19 * r42);
    r47 = r36 + r47;
    r47 = fmaf(r13, r47, r15 * r20);
    r15 = r5 * r47;
    r36 = r10 * r69;
    r36 = r36 * r41;
    r36 = fmaf(r40, r36, r12 * r15);
    r15 = fmaf(r1, r36, r0 * r19);
    r38 = fmaf(r60, r15, r49);
    r63 = r21 * r9;
    r36 = fmaf(r3, r36, r2 * r19);
    r15 = fmaf(r15, r70, r36 * r63);
    r63 = r15 * r43;
    r63 = r8 <= r51 ? r15 : r63;
    r19 = r15 * r61;
    r63 = r45 < r52 ? r19 : r63;
    r19 = r15 * r56;
    r63 = r45 < r54 ? r19 : r63;
    r63 = r45 < r50 ? r15 : r63;
    r19 = r50 * r63;
    r15 = fmaf(r15, r57, r58 * r19);
    r15 = r50 * r15;
    r15 = r15 * r59;
    r15 = r8 <= r34 ? r48 : r15;
    r38 = fmaf(r44, r15, r38);
    r36 = fmaf(r60, r36, r71);
    r36 = fmaf(r9, r15, r36);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r7,
                                          r22,
                                          r38,
                                          r36);
    r62 = r31 + r62;
    r62 = r62 + r65;
    r62 = r62 + r68;
    r62 = fmaf(r14, r62, r13 * r25);
    r23 = r27 + r23;
    r23 = fmaf(r14, r23, r13 * r37);
    r62 = fmaf(r23, r4, r62 * r42);
    r27 = r10 * r23;
    r27 = r27 * r41;
    r24 = r17 + r24;
    r18 = r66 + r18;
    r18 = fmaf(r13, r18, r14 * r24);
    r13 = r5 * r18;
    r13 = fmaf(r12, r13, r40 * r27);
    r27 = fmaf(r1, r13, r0 * r62);
    r24 = fmaf(r60, r27, r49);
    r14 = r21 * r9;
    r13 = fmaf(r3, r13, r2 * r62);
    r27 = fmaf(r27, r70, r13 * r14);
    r14 = r27 * r43;
    r14 = r8 <= r51 ? r27 : r14;
    r62 = r27 * r61;
    r14 = r45 < r52 ? r62 : r14;
    r62 = r27 * r56;
    r14 = r45 < r54 ? r62 : r14;
    r14 = r45 < r50 ? r27 : r14;
    r62 = r50 * r14;
    r27 = fmaf(r27, r57, r58 * r62);
    r27 = r50 * r27;
    r27 = r27 * r59;
    r27 = r8 <= r34 ? r48 : r27;
    r24 = fmaf(r44, r27, r24);
    r13 = fmaf(r60, r13, r71);
    r13 = fmaf(r9, r27, r13);
    r27 = r0 * r21;
    r27 = r27 * r44;
    r62 = r21 * r9;
    r66 = r2 * r42;
    r62 = fmaf(r66, r62, r42 * r27);
    r17 = r62 * r43;
    r17 = r8 <= r51 ? r62 : r17;
    r68 = r62 * r61;
    r17 = r45 < r52 ? r68 : r17;
    r68 = r62 * r56;
    r17 = r45 < r54 ? r68 : r17;
    r17 = r45 < r50 ? r62 : r17;
    r68 = r50 * r17;
    r68 = fmaf(r58, r68, r62 * r57);
    r68 = r50 * r68;
    r68 = r68 * r59;
    r68 = r8 <= r34 ? r48 : r68;
    r62 = fmaf(r44, r68, r49);
    r65 = r0 * r60;
    r62 = fmaf(r42, r65, r62);
    r68 = fmaf(r9, r68, r71);
    r68 = fmaf(r60, r66, r68);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r24,
                                          r13,
                                          r62,
                                          r68);
    r66 = r21 * r9;
    r65 = r3 * r10;
    r65 = r65 * r41;
    r65 = fmaf(r40, r65, r2 * r4);
    r31 = r1 * r10;
    r31 = r31 * r41;
    r31 = fmaf(r40, r31, r0 * r4);
    r66 = fmaf(r31, r70, r65 * r66);
    r15 = r66 * r43;
    r15 = r8 <= r51 ? r66 : r15;
    r19 = r66 * r61;
    r15 = r45 < r52 ? r19 : r15;
    r19 = r66 * r56;
    r15 = r45 < r54 ? r19 : r15;
    r15 = r45 < r50 ? r66 : r15;
    r19 = r50 * r15;
    r66 = fmaf(r66, r57, r58 * r19);
    r66 = r50 * r66;
    r66 = r66 * r59;
    r66 = r8 <= r34 ? r48 : r66;
    r19 = fmaf(r44, r66, r49);
    r19 = fmaf(r60, r31, r19);
    r65 = fmaf(r60, r65, r71);
    r65 = fmaf(r9, r66, r65);
    r66 = r5 * r1;
    r66 = r66 * r12;
    r31 = r5 * r3;
    r31 = r31 * r21;
    r31 = r31 * r9;
    r31 = fmaf(r12, r31, r70 * r66);
    r66 = r31 * r43;
    r66 = r8 <= r51 ? r31 : r66;
    r39 = r31 * r61;
    r66 = r45 < r52 ? r39 : r66;
    r39 = r31 * r56;
    r66 = r45 < r54 ? r39 : r66;
    r66 = r45 < r50 ? r31 : r66;
    r39 = r50 * r66;
    r39 = fmaf(r58, r39, r31 * r57);
    r39 = r50 * r39;
    r39 = r39 * r59;
    r39 = r8 <= r34 ? r48 : r39;
    r31 = fmaf(r44, r39, r49);
    r16 = r5 * r1;
    r16 = r16 * r60;
    r31 = fmaf(r12, r16, r31);
    r39 = fmaf(r9, r39, r71);
    r16 = r5 * r3;
    r16 = r16 * r60;
    r39 = fmaf(r12, r16, r39);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r31,
                                          r39,
                                          r19,
                                          r65);
    r16 = r10 * r60;
    r30 = r9 * r16;
    r55 = r44 * r7;
    r55 = fmaf(r16, r55, r22 * r30);
    r72 = r44 * r38;
    r72 = fmaf(r36, r30, r16 * r72);
    r73 = r44 * r24;
    r73 = fmaf(r16, r73, r13 * r30);
    r74 = r44 * r62;
    r74 = fmaf(r16, r74, r68 * r30);
    WriteSum4<float, float>((float*)inout_shared, r55, r72, r73, r74);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = r44 * r31;
    r74 = fmaf(r39, r30, r16 * r74);
    r73 = r44 * r19;
    r73 = fmaf(r16, r73, r65 * r30);
    WriteSum2<float, float>((float*)inout_shared, r74, r73);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r7, r7, r22 * r22);
    r74 = fmaf(r36, r36, r38 * r38);
    r72 = fmaf(r13, r13, r24 * r24);
    r55 = fmaf(r62, r62, r68 * r68);
    WriteSum4<float, float>((float*)inout_shared, r73, r74, r72, r55);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r39, r39, r31 * r31);
    r72 = fmaf(r65, r65, r19 * r19);
    WriteSum2<float, float>((float*)inout_shared, r55, r72);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fmaf(r22, r36, r7 * r38);
    r55 = fmaf(r7, r24, r22 * r13);
    r74 = fmaf(r7, r62, r22 * r68);
    r73 = fmaf(r7, r31, r22 * r39);
    WriteSum4<float, float>((float*)inout_shared, r72, r55, r74, r73);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fmaf(r22, r65, r7 * r19);
    r73 = fmaf(r38, r24, r36 * r13);
    r74 = fmaf(r36, r68, r38 * r62);
    r55 = fmaf(r38, r31, r36 * r39);
    WriteSum4<float, float>((float*)inout_shared, r22, r73, r74, r55);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r36, r65, r38 * r19);
    r55 = fmaf(r13, r68, r24 * r62);
    r74 = fmaf(r13, r39, r24 * r31);
    r13 = fmaf(r13, r65, r24 * r19);
    WriteSum4<float, float>((float*)inout_shared, r36, r55, r74, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r62, r31, r68 * r39);
    r68 = fmaf(r62, r19, r68 * r65);
    r65 = fmaf(r31, r19, r39 * r65);
    WriteSum3<float, float>((float*)inout_shared, r13, r68, r65);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r6 * r12;
    r68 = r2 * r21;
    r68 = r68 * r6;
    r68 = r68 * r9;
    r68 = fmaf(r12, r68, r27 * r65);
    r65 = r68 * r43;
    r65 = r8 <= r51 ? r68 : r65;
    r13 = r68 * r61;
    r65 = r45 < r52 ? r13 : r65;
    r13 = r68 * r56;
    r65 = r45 < r54 ? r13 : r65;
    r65 = r45 < r50 ? r68 : r65;
    r13 = r50 * r65;
    r13 = fmaf(r58, r13, r68 * r57);
    r13 = r50 * r13;
    r13 = r13 * r59;
    r13 = r8 <= r34 ? r48 : r13;
    r68 = fmaf(r44, r13, r49);
    r39 = r0 * r6;
    r39 = r39 * r60;
    r68 = fmaf(r12, r39, r68);
    r13 = fmaf(r9, r13, r71);
    r39 = r2 * r6;
    r39 = r39 * r60;
    r13 = fmaf(r12, r39, r13);
    r39 = r1 * r11;
    r39 = r39 * r12;
    r74 = r3 * r21;
    r74 = r74 * r11;
    r74 = r74 * r9;
    r74 = fmaf(r12, r74, r70 * r39);
    r39 = r74 * r43;
    r39 = r8 <= r51 ? r74 : r39;
    r55 = r74 * r61;
    r39 = r45 < r52 ? r55 : r39;
    r55 = r74 * r56;
    r39 = r45 < r54 ? r55 : r39;
    r39 = r45 < r50 ? r74 : r39;
    r55 = r50 * r39;
    r74 = fmaf(r74, r57, r58 * r55);
    r74 = r50 * r74;
    r74 = r74 * r59;
    r74 = r8 <= r34 ? r48 : r74;
    r55 = fmaf(r44, r74, r49);
    r36 = r1 * r11;
    r36 = r36 * r60;
    r55 = fmaf(r12, r36, r55);
    r74 = fmaf(r9, r74, r71);
    r36 = r3 * r11;
    r36 = r36 * r60;
    r74 = fmaf(r12, r36, r74);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r68,
                                          r13,
                                          r55,
                                          r74);
    r36 = fmaf(r0, r60, r49);
    r73 = r2 * r21;
    r73 = fmaf(r9, r73, r27);
    r27 = r73 * r43;
    r27 = r8 <= r51 ? r73 : r27;
    r22 = r73 * r61;
    r27 = r45 < r52 ? r22 : r27;
    r22 = r73 * r56;
    r27 = r45 < r54 ? r22 : r27;
    r27 = r45 < r50 ? r73 : r27;
    r22 = r50 * r27;
    r22 = fmaf(r58, r22, r73 * r57);
    r22 = r50 * r22;
    r22 = r22 * r59;
    r22 = r8 <= r34 ? r48 : r22;
    r36 = fmaf(r44, r22, r36);
    r73 = fmaf(r2, r60, r71);
    r73 = fmaf(r9, r22, r73);
    r22 = r3 * r21;
    r22 = fmaf(r1, r70, r9 * r22);
    r72 = r22 * r43;
    r72 = r8 <= r51 ? r22 : r72;
    r75 = r22 * r61;
    r72 = r45 < r52 ? r75 : r72;
    r75 = r22 * r56;
    r72 = r45 < r54 ? r75 : r72;
    r72 = r45 < r50 ? r22 : r72;
    r75 = r50 * r72;
    r22 = fmaf(r22, r57, r58 * r75);
    r22 = r50 * r22;
    r22 = r22 * r59;
    r22 = r8 <= r34 ? r48 : r22;
    r75 = fmaf(r44, r22, r49);
    r75 = fmaf(r1, r60, r75);
    r22 = fmaf(r9, r22, r71);
    r22 = fmaf(r3, r60, r22);
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          4 * out_calib_jac_num_alloc,
                                          global_thread_idx,
                                          r36,
                                          r73,
                                          r75,
                                          r22);
    r76 = r44 * r68;
    r76 = fmaf(r16, r76, r13 * r30);
    r77 = r44 * r55;
    r77 = fmaf(r74, r30, r16 * r77);
    r78 = r44 * r36;
    r78 = fmaf(r16, r78, r73 * r30);
    r79 = r44 * r75;
    r79 = fmaf(r16, r79, r22 * r30);
    WriteSum4<float, float>((float*)inout_shared, r76, r77, r78, r79);
  };
  FlushSumShared<4, float>(out_calib_njtr,
                           0 * out_calib_njtr_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fmaf(r13, r13, r68 * r68);
    r78 = fmaf(r74, r74, r55 * r55);
    r77 = fmaf(r73, r73, r36 * r36);
    r76 = fmaf(r75, r75, r22 * r22);
    WriteSum4<float, float>((float*)inout_shared, r79, r78, r77, r76);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fmaf(r68, r55, r13 * r74);
    r77 = fmaf(r68, r36, r13 * r73);
    r13 = fmaf(r68, r75, r13 * r22);
    r78 = fmaf(r74, r73, r55 * r36);
    WriteSum4<float, float>((float*)inout_shared, r76, r77, r13, r78);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r55, r75, r74 * r22);
    r22 = fmaf(r36, r75, r73 * r22);
    WriteSum2<float, float>((float*)inout_shared, r74, r22);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r21 * r9;
    r33 = fmaf(r26, r4, r33 * r42);
    r74 = r5 * r20;
    r73 = r26 * r10;
    r73 = r73 * r41;
    r73 = fmaf(r40, r73, r12 * r74);
    r74 = fmaf(r3, r73, r2 * r33);
    r73 = fmaf(r1, r73, r0 * r33);
    r22 = fmaf(r73, r70, r74 * r22);
    r33 = r22 * r43;
    r33 = r8 <= r51 ? r22 : r33;
    r78 = r22 * r61;
    r33 = r45 < r52 ? r78 : r33;
    r78 = r22 * r56;
    r33 = r45 < r54 ? r78 : r33;
    r33 = r45 < r50 ? r22 : r33;
    r78 = r50 * r33;
    r78 = fmaf(r58, r78, r22 * r57);
    r78 = r50 * r78;
    r78 = r78 * r59;
    r78 = r8 <= r34 ? r48 : r78;
    r22 = fmaf(r44, r78, r49);
    r22 = fmaf(r60, r73, r22);
    r78 = fmaf(r9, r78, r71);
    r78 = fmaf(r60, r74, r78);
    r74 = r21 * r9;
    r25 = fmaf(r37, r4, r25 * r42);
    r73 = r5 * r29;
    r13 = r37 * r10;
    r13 = r13 * r41;
    r13 = fmaf(r40, r13, r12 * r73);
    r73 = fmaf(r3, r13, r2 * r25);
    r13 = fmaf(r1, r13, r0 * r25);
    r74 = fmaf(r13, r70, r73 * r74);
    r25 = r74 * r43;
    r25 = r8 <= r51 ? r74 : r25;
    r77 = r74 * r61;
    r25 = r45 < r52 ? r77 : r25;
    r77 = r74 * r56;
    r25 = r45 < r54 ? r77 : r25;
    r25 = r45 < r50 ? r74 : r25;
    r77 = r50 * r25;
    r74 = fmaf(r74, r57, r58 * r77);
    r74 = r50 * r74;
    r74 = r74 * r59;
    r74 = r8 <= r34 ? r48 : r74;
    r77 = fmaf(r44, r74, r49);
    r77 = fmaf(r60, r13, r77);
    r73 = fmaf(r60, r73, r71);
    r73 = fmaf(r9, r74, r73);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r22,
                                          r78,
                                          r77,
                                          r73);
    r74 = r21 * r9;
    r4 = fmaf(r32, r4, r28 * r42);
    r42 = r5 * r35;
    r28 = r32 * r10;
    r28 = r28 * r41;
    r28 = fmaf(r40, r28, r12 * r42);
    r42 = fmaf(r3, r28, r2 * r4);
    r28 = fmaf(r1, r28, r0 * r4);
    r70 = fmaf(r28, r70, r42 * r74);
    r43 = r70 * r43;
    r43 = r8 <= r51 ? r70 : r43;
    r61 = r70 * r61;
    r43 = r45 < r52 ? r61 : r43;
    r56 = r70 * r56;
    r43 = r45 < r54 ? r56 : r43;
    r43 = r45 < r50 ? r70 : r43;
    r45 = r50 * r43;
    r57 = fmaf(r70, r57, r58 * r45);
    r57 = r50 * r57;
    r57 = r57 * r59;
    r57 = r8 <= r34 ? r48 : r57;
    r49 = fmaf(r44, r57, r49);
    r49 = fmaf(r60, r28, r49);
    r42 = fmaf(r60, r42, r71);
    r42 = fmaf(r9, r57, r42);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r49,
                                          r42);
    r57 = r44 * r22;
    r57 = fmaf(r78, r30, r16 * r57);
    r71 = r44 * r77;
    r71 = fmaf(r73, r30, r16 * r71);
    r28 = r44 * r49;
    r30 = fmaf(r42, r30, r16 * r28);
    WriteSum3<float, float>((float*)inout_shared, r57, r71, r30);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r30 = fmaf(r78, r78, r22 * r22);
    r71 = fmaf(r73, r73, r77 * r77);
    r57 = fmaf(r49, r49, r42 * r42);
    WriteSum3<float, float>((float*)inout_shared, r30, r71, r57);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fmaf(r22, r77, r78 * r73);
    r78 = fmaf(r78, r42, r22 * r49);
    r42 = fmaf(r77, r49, r73 * r42);
    WriteSum3<float, float>((float*)inout_shared, r57, r78, r42);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void PinholeResJac(float* pose,
                   unsigned int pose_num_alloc,
                   SharedIndex* pose_indices,
                   float* calib,
                   unsigned int calib_num_alloc,
                   SharedIndex* calib_indices,
                   float* point,
                   unsigned int point_num_alloc,
                   SharedIndex* point_indices,
                   float* pixel,
                   unsigned int pixel_num_alloc,
                   float* weight_loss,
                   unsigned int weight_loss_num_alloc,
                   float* out_res,
                   unsigned int out_res_num_alloc,
                   float* out_pose_jac,
                   unsigned int out_pose_jac_num_alloc,
                   float* const out_pose_njtr,
                   unsigned int out_pose_njtr_num_alloc,
                   float* const out_pose_precond_diag,
                   unsigned int out_pose_precond_diag_num_alloc,
                   float* const out_pose_precond_tril,
                   unsigned int out_pose_precond_tril_num_alloc,
                   float* out_calib_jac,
                   unsigned int out_calib_jac_num_alloc,
                   float* const out_calib_njtr,
                   unsigned int out_calib_njtr_num_alloc,
                   float* const out_calib_precond_diag,
                   unsigned int out_calib_precond_diag_num_alloc,
                   float* const out_calib_precond_tril,
                   unsigned int out_calib_precond_tril_num_alloc,
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
  PinholeResJacKernel<<<n_blocks, 1024>>>(pose,
                                          pose_num_alloc,
                                          pose_indices,
                                          calib,
                                          calib_num_alloc,
                                          calib_indices,
                                          point,
                                          point_num_alloc,
                                          point_indices,
                                          pixel,
                                          pixel_num_alloc,
                                          weight_loss,
                                          weight_loss_num_alloc,
                                          out_res,
                                          out_res_num_alloc,
                                          out_pose_jac,
                                          out_pose_jac_num_alloc,
                                          out_pose_njtr,
                                          out_pose_njtr_num_alloc,
                                          out_pose_precond_diag,
                                          out_pose_precond_diag_num_alloc,
                                          out_pose_precond_tril,
                                          out_pose_precond_tril_num_alloc,
                                          out_calib_jac,
                                          out_calib_jac_num_alloc,
                                          out_calib_njtr,
                                          out_calib_njtr_num_alloc,
                                          out_calib_precond_diag,
                                          out_calib_precond_diag_num_alloc,
                                          out_calib_precond_tril,
                                          out_calib_precond_tril_num_alloc,
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