#include "kernel_pinhole_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeResJacFirstKernel(double* pose,
                             unsigned int pose_num_alloc,
                             SharedIndex* pose_indices,
                             double* calib,
                             unsigned int calib_num_alloc,
                             SharedIndex* calib_indices,
                             double* point,
                             unsigned int point_num_alloc,
                             SharedIndex* point_indices,
                             double* pixel,
                             unsigned int pixel_num_alloc,
                             double* weight_loss,
                             unsigned int weight_loss_num_alloc,
                             double* out_res,
                             unsigned int out_res_num_alloc,
                             double* const out_rTr,
                             double* out_pose_jac,
                             unsigned int out_pose_jac_num_alloc,
                             double* const out_pose_njtr,
                             unsigned int out_pose_njtr_num_alloc,
                             double* const out_pose_precond_diag,
                             unsigned int out_pose_precond_diag_num_alloc,
                             double* const out_pose_precond_tril,
                             unsigned int out_pose_precond_tril_num_alloc,
                             double* out_calib_jac,
                             unsigned int out_calib_jac_num_alloc,
                             double* const out_calib_njtr,
                             unsigned int out_calib_njtr_num_alloc,
                             double* const out_calib_precond_diag,
                             unsigned int out_calib_precond_diag_num_alloc,
                             double* const out_calib_precond_tril,
                             unsigned int out_calib_precond_tril_num_alloc,
                             double* out_point_jac,
                             unsigned int out_point_jac_num_alloc,
                             double* const out_point_njtr,
                             unsigned int out_point_njtr_num_alloc,
                             double* const out_point_precond_diag,
                             unsigned int out_point_precond_diag_num_alloc,
                             double* const out_point_precond_tril,
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

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78;

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
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r2, r7);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      point, 0 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r10, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r12 = r10 * r11;
    r13 = 2.00000000000000000e+00;
    r12 = r12 * r13;
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = -2.00000000000000000e+00;
    r17 = r14 * r16;
    r18 = r15 * r17;
    r19 = r12 + r18;
    r2 = fma(r9, r19, r2);
  };
  LoadShared<1, double, double>(
      point, 2 * point_num_alloc, point_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_indices_loc[threadIdx.x].target, r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r21 = r10 * r14;
    r21 = r21 * r13;
    r22 = r11 * r15;
    r22 = r22 * r13;
    r23 = r21 + r22;
    r24 = r14 * r17;
    r25 = 1.00000000000000000e+00;
    r26 = r11 * r11;
    r27 = fma(r16, r26, r25);
    r28 = r24 + r27;
    r2 = fma(r20, r23, r2);
    r2 = fma(r8, r28, r2);
  };
  LoadShared<2, double, double>(
      calib, 0 * calib_num_alloc, calib_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, calib_indices_loc[threadIdx.x].target, r29, r30);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r31 = 1.00000000000000008e-15;
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r11 * r14;
    r33 = r33 * r13;
    r34 = r10 * r15;
    r34 = r34 * r13;
    r35 = r33 + r34;
    r32 = fma(r9, r35, r32);
    r36 = r11 * r15;
    r36 = r36 * r16;
    r21 = r21 + r36;
    r37 = r10 * r10;
    r38 = r16 * r37;
    r27 = r38 + r27;
    r32 = fma(r8, r21, r32);
    r32 = fma(r20, r27, r32);
    r39 = copysign(1.0, r32);
    r39 = fma(r31, r39, r32);
    r32 = 1.0 / r39;
    r40 = r29 * r32;
    r4 = fma(r2, r40, r4);
    r5 = fma(r5, r6, r3);
    r3 = r14 * r15;
    r3 = r3 * r13;
    r12 = r12 + r3;
    r7 = fma(r8, r12, r7);
    r41 = r10 * r15;
    r41 = r41 * r16;
    r33 = r33 + r41;
    r24 = r25 + r24;
    r24 = r24 + r38;
    r7 = fma(r20, r33, r7);
    r7 = fma(r9, r24, r7);
    r38 = r30 * r7;
    r5 = fma(r32, r38, r5);
    r42 = fma(r1, r5, r0 * r4);
    ReadIdx1<1024, double, double, double>(
        weight_loss, 6 * weight_loss_num_alloc, global_thread_idx, r43);
    r44 = 0.00000000000000000e+00;
    r43 = fmax(r43, r44);
    r45 = sqrt(r43);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 2 * weight_loss_num_alloc, global_thread_idx, r46, r47);
    r5 = fma(r47, r5, r46 * r4);
    r4 = fma(r42, r42, r5 * r5);
    ReadIdx2<1024, double, double, double2>(
        weight_loss, 4 * weight_loss_num_alloc, global_thread_idx, r48, r49);
    r50 = 5.00000000000000000e-01;
    r49 = fmax(r49, r31);
    r51 = r49 * r49;
    r52 = r13 * r49;
    r53 = fmax(r31, r4);
    r54 = sqrt(r53);
    r55 = r6 * r49;
    r55 = fma(r49, r55, r54 * r52);
    r55 = r4 <= r51 ? r4 : r55;
    r52 = 2.50000000000000000e+00;
    r54 = r49 * r49;
    r56 = 1.0 / r51;
    r56 = fma(r4, r56, r25);
    r57 = log(r56);
    r54 = r54 * r57;
    r55 = r48 < r52 ? r54 : r55;
    r54 = 1.50000000000000000e+00;
    r57 = r13 * r49;
    r58 = sqrt(r56);
    r58 = r6 + r58;
    r57 = r57 * r49;
    r57 = r57 * r58;
    r55 = r48 < r54 ? r57 : r55;
    r55 = r48 < r50 ? r4 : r55;
    r57 = fmax(r44, r55);
    r58 = 1.0 / r53;
    r58 = r43 * r58;
    r59 = r57 * r58;
    r60 = sqrt(r59);
    r60 = r4 <= r31 ? r45 : r60;
    r45 = r42 * r60;
    r61 = r5 * r60;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r45, r61);
    r61 = r42 * r42;
    r61 = r61 * r60;
    r45 = r5 * r5;
    r45 = r45 * r60;
    r45 = fma(r60, r45, r60 * r61);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r45);
  if (global_thread_idx < problem_size) {
    r45 = r6 * r42;
    r61 = 2.50000000000000000e-01;
    r62 = r4 <= r51 ? r44 : r44;
    r62 = r48 < r52 ? r44 : r62;
    r62 = r48 < r54 ? r44 : r62;
    r62 = r48 < r50 ? r44 : r62;
    r62 = r61 * r62;
    r59 = rsqrt(r59);
    r55 = copysign(1.0, r55);
    r55 = r25 + r55;
    r58 = r55 * r58;
    r62 = r62 * r59;
    r62 = r62 * r58;
    r62 = r4 <= r31 ? r44 : r62;
    r45 = r45 * r62;
    r55 = r13 * r42;
    r61 = r10 * r11;
    r61 = r61 * r16;
    r3 = r3 + r61;
    r3 = fma(r9, r23, r20 * r3);
    r11 = r11 * r17;
    r41 = r41 + r11;
    r16 = r14 * r14;
    r63 = r6 * r26;
    r64 = r16 + r63;
    r15 = r15 * r15;
    r65 = r6 * r37;
    r66 = r15 + r65;
    r67 = r64 + r66;
    r67 = fma(r9, r67, r20 * r41);
    r29 = r29 * r2;
    r39 = r39 * r39;
    r39 = 1.0 / r39;
    r29 = r29 * r6;
    r29 = r29 * r39;
    r3 = fma(r67, r29, r3 * r40);
    r41 = r6 * r67;
    r41 = r41 * r39;
    r68 = r6 * r15;
    r69 = r37 + r68;
    r64 = r64 + r69;
    r64 = fma(r20, r64, r9 * r33);
    r70 = r30 * r64;
    r70 = fma(r32, r70, r38 * r41);
    r41 = fma(r1, r70, r0 * r3);
    r70 = fma(r47, r70, r46 * r3);
    r3 = r13 * r5;
    r55 = fma(r70, r3, r41 * r55);
    r71 = r50 * r49;
    r72 = -1.00000000000000008e-15;
    r72 = r72 + r4;
    r72 = copysign(1.0, r72);
    r72 = r25 + r72;
    r25 = rsqrt(r53);
    r71 = r71 * r72;
    r71 = r71 * r25;
    r25 = r55 * r71;
    r25 = r4 <= r51 ? r55 : r25;
    r73 = 1.0 / r56;
    r74 = r55 * r73;
    r25 = r48 < r52 ? r74 : r25;
    r56 = rsqrt(r56);
    r74 = r55 * r56;
    r25 = r48 < r54 ? r74 : r25;
    r25 = r48 < r50 ? r55 : r25;
    r74 = r50 * r25;
    r57 = r43 * r57;
    r43 = -5.00000000000000000e-01;
    r53 = r53 * r53;
    r53 = 1.0 / r53;
    r57 = r57 * r43;
    r57 = r57 * r72;
    r57 = r57 * r53;
    r55 = fma(r55, r57, r58 * r74);
    r55 = r50 * r55;
    r55 = r55 * r59;
    r55 = r4 <= r31 ? r44 : r55;
    r74 = fma(r42, r55, r45);
    r74 = fma(r60, r41, r74);
    r41 = r6 * r5;
    r41 = r41 * r62;
    r55 = fma(r5, r55, r41);
    r55 = fma(r60, r70, r55);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r74, r55);
    r70 = r13 * r42;
    r14 = r14 * r14;
    r14 = r14 * r6;
    r62 = r26 + r14;
    r69 = r69 + r62;
    r69 = fma(r8, r69, r20 * r21);
    r17 = r10 * r17;
    r36 = r36 + r17;
    r15 = r37 + r15;
    r15 = r15 + r63;
    r15 = r15 + r14;
    r15 = fma(r20, r15, r8 * r36);
    r15 = fma(r15, r40, r69 * r29);
    r11 = r34 + r11;
    r11 = fma(r8, r11, r20 * r12);
    r20 = r30 * r11;
    r34 = r6 * r69;
    r34 = r34 * r39;
    r34 = fma(r38, r34, r32 * r20);
    r20 = fma(r1, r34, r0 * r15);
    r34 = fma(r47, r34, r46 * r15);
    r70 = fma(r34, r3, r20 * r70);
    r15 = r70 * r71;
    r15 = r4 <= r51 ? r70 : r15;
    r36 = r70 * r73;
    r15 = r48 < r52 ? r36 : r15;
    r36 = r70 * r56;
    r15 = r48 < r54 ? r36 : r15;
    r15 = r48 < r50 ? r70 : r15;
    r36 = r50 * r15;
    r70 = fma(r70, r57, r58 * r36);
    r70 = r50 * r70;
    r70 = r70 * r59;
    r70 = r4 <= r31 ? r44 : r70;
    r36 = fma(r42, r70, r45);
    r36 = fma(r60, r20, r36);
    r34 = fma(r60, r34, r41);
    r34 = fma(r5, r70, r34);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 2 * out_pose_jac_num_alloc, global_thread_idx, r36, r34);
    r16 = r26 + r16;
    r16 = r16 + r65;
    r16 = r16 + r68;
    r16 = fma(r9, r16, r8 * r19);
    r17 = r22 + r17;
    r17 = fma(r9, r17, r8 * r35);
    r16 = fma(r17, r29, r16 * r40);
    r18 = r61 + r18;
    r62 = r66 + r62;
    r62 = fma(r8, r62, r9 * r18);
    r8 = r30 * r62;
    r18 = r6 * r17;
    r18 = r18 * r39;
    r18 = fma(r38, r18, r32 * r8);
    r8 = fma(r1, r18, r0 * r16);
    r9 = fma(r60, r8, r45);
    r66 = r13 * r42;
    r18 = fma(r47, r18, r46 * r16);
    r66 = fma(r18, r3, r8 * r66);
    r8 = r66 * r71;
    r8 = r4 <= r51 ? r66 : r8;
    r16 = r66 * r73;
    r8 = r48 < r52 ? r16 : r8;
    r16 = r66 * r56;
    r8 = r48 < r54 ? r16 : r8;
    r8 = r48 < r50 ? r66 : r8;
    r16 = r50 * r8;
    r16 = fma(r58, r16, r66 * r57);
    r16 = r50 * r16;
    r16 = r16 * r59;
    r16 = r4 <= r31 ? r44 : r16;
    r9 = fma(r42, r16, r9);
    r18 = fma(r60, r18, r41);
    r18 = fma(r5, r16, r18);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r9, r18);
    r16 = r46 * r13;
    r16 = r16 * r5;
    r66 = r13 * r42;
    r61 = r0 * r40;
    r66 = fma(r61, r66, r40 * r16);
    r22 = r66 * r71;
    r22 = r4 <= r51 ? r66 : r22;
    r68 = r66 * r73;
    r22 = r48 < r52 ? r68 : r22;
    r68 = r66 * r56;
    r22 = r48 < r54 ? r68 : r22;
    r22 = r48 < r50 ? r66 : r22;
    r68 = r50 * r22;
    r68 = fma(r58, r68, r66 * r57);
    r68 = r50 * r68;
    r68 = r68 * r59;
    r68 = r4 <= r31 ? r44 : r68;
    r66 = fma(r42, r68, r45);
    r66 = fma(r60, r61, r66);
    r68 = fma(r5, r68, r41);
    r61 = r46 * r60;
    r68 = fma(r40, r61, r68);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 6 * out_pose_jac_num_alloc, global_thread_idx, r66, r68);
    r61 = r30 * r47;
    r61 = r61 * r32;
    r65 = r30 * r1;
    r65 = r65 * r13;
    r65 = r65 * r42;
    r65 = fma(r32, r65, r3 * r61);
    r61 = r65 * r71;
    r61 = r4 <= r51 ? r65 : r61;
    r26 = r65 * r73;
    r61 = r48 < r52 ? r26 : r61;
    r26 = r65 * r56;
    r61 = r48 < r54 ? r26 : r61;
    r61 = r48 < r50 ? r65 : r61;
    r26 = r50 * r61;
    r65 = fma(r65, r57, r58 * r26);
    r65 = r50 * r65;
    r65 = r65 * r59;
    r65 = r4 <= r31 ? r44 : r65;
    r26 = fma(r42, r65, r45);
    r70 = r30 * r1;
    r70 = r70 * r60;
    r26 = fma(r32, r70, r26);
    r65 = fma(r5, r65, r41);
    r70 = r30 * r47;
    r70 = r70 * r60;
    r65 = fma(r32, r70, r65);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r26, r65);
    r70 = r1 * r6;
    r70 = r70 * r39;
    r70 = fma(r0, r29, r38 * r70);
    r20 = fma(r60, r70, r45);
    r14 = r13 * r42;
    r63 = r47 * r6;
    r63 = r63 * r39;
    r63 = fma(r46, r29, r38 * r63);
    r14 = fma(r63, r3, r70 * r14);
    r70 = r14 * r71;
    r70 = r4 <= r51 ? r14 : r70;
    r37 = r14 * r73;
    r70 = r48 < r52 ? r37 : r70;
    r37 = r14 * r56;
    r70 = r48 < r54 ? r37 : r70;
    r70 = r48 < r50 ? r14 : r70;
    r37 = r50 * r70;
    r14 = fma(r14, r57, r58 * r37);
    r14 = r50 * r14;
    r14 = r14 * r59;
    r14 = r4 <= r31 ? r44 : r14;
    r20 = fma(r42, r14, r20);
    r14 = fma(r5, r14, r41);
    r14 = fma(r60, r63, r14);
    WriteIdx2<1024, double, double, double2>(
        out_pose_jac, 10 * out_pose_jac_num_alloc, global_thread_idx, r20, r14);
    r63 = r6 * r60;
    r37 = r42 * r63;
    r10 = r5 * r55;
    r10 = fma(r63, r10, r74 * r37);
    r53 = r5 * r34;
    r53 = fma(r36, r37, r63 * r53);
    WriteSum2<double, double>((double*)inout_shared, r10, r53);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = r5 * r18;
    r53 = fma(r9, r37, r63 * r53);
    r10 = r5 * r68;
    r10 = fma(r63, r10, r66 * r37);
    WriteSum2<double, double>((double*)inout_shared, r53, r10);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r5 * r65;
    r10 = fma(r26, r37, r63 * r10);
    r53 = r5 * r14;
    r53 = fma(r20, r37, r63 * r53);
    WriteSum2<double, double>((double*)inout_shared, r10, r53);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fma(r74, r74, r55 * r55);
    r10 = fma(r34, r34, r36 * r36);
    WriteSum2<double, double>((double*)inout_shared, r53, r10);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r9, r9, r18 * r18);
    r53 = fma(r68, r68, r66 * r66);
    WriteSum2<double, double>((double*)inout_shared, r10, r53);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fma(r65, r65, r26 * r26);
    r10 = fma(r14, r14, r20 * r20);
    WriteSum2<double, double>((double*)inout_shared, r53, r10);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r74, r36, r55 * r34);
    r53 = fma(r74, r9, r55 * r18);
    WriteSum2<double, double>((double*)inout_shared, r10, r53);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fma(r55, r68, r74 * r66);
    r10 = fma(r55, r65, r74 * r26);
    WriteSum2<double, double>((double*)inout_shared, r53, r10);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r74, r20, r55 * r14);
    r10 = fma(r34, r18, r36 * r9);
    WriteSum2<double, double>((double*)inout_shared, r74, r10);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fma(r36, r66, r34 * r68);
    r74 = fma(r36, r26, r34 * r65);
    WriteSum2<double, double>((double*)inout_shared, r10, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fma(r34, r14, r36 * r20);
    r74 = fma(r18, r68, r9 * r66);
    WriteSum2<double, double>((double*)inout_shared, r36, r74);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r9, r26, r18 * r65);
    r9 = fma(r9, r20, r18 * r14);
    WriteSum2<double, double>((double*)inout_shared, r74, r9);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r66, r26, r68 * r65);
    r66 = fma(r68, r14, r66 * r20);
    WriteSum2<double, double>((double*)inout_shared, r9, r66);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fma(r65, r14, r26 * r20);
    WriteSum1<double, double>((double*)inout_shared, r20);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = r0 * r13;
    r20 = r20 * r2;
    r20 = r20 * r42;
    r26 = r2 * r32;
    r26 = fma(r16, r26, r32 * r20);
    r20 = r26 * r71;
    r20 = r4 <= r51 ? r26 : r20;
    r66 = r26 * r73;
    r20 = r48 < r52 ? r66 : r20;
    r66 = r26 * r56;
    r20 = r48 < r54 ? r66 : r20;
    r20 = r48 < r50 ? r26 : r20;
    r66 = r50 * r20;
    r26 = fma(r26, r57, r58 * r66);
    r26 = r50 * r26;
    r26 = r26 * r59;
    r26 = r4 <= r31 ? r44 : r26;
    r66 = fma(r42, r26, r45);
    r9 = r0 * r2;
    r9 = r9 * r60;
    r66 = fma(r32, r9, r66);
    r26 = fma(r5, r26, r41);
    r9 = r46 * r2;
    r9 = r9 * r60;
    r26 = fma(r32, r9, r26);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             0 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r66,
                                             r26);
    r9 = r1 * r13;
    r9 = r9 * r7;
    r9 = r9 * r42;
    r74 = r47 * r7;
    r74 = r74 * r32;
    r74 = fma(r3, r74, r32 * r9);
    r9 = r74 * r71;
    r9 = r4 <= r51 ? r74 : r9;
    r36 = r74 * r73;
    r9 = r48 < r52 ? r36 : r9;
    r36 = r74 * r56;
    r9 = r48 < r54 ? r36 : r9;
    r9 = r48 < r50 ? r74 : r9;
    r36 = r50 * r9;
    r36 = fma(r58, r36, r74 * r57);
    r36 = r50 * r36;
    r36 = r36 * r59;
    r36 = r4 <= r31 ? r44 : r36;
    r74 = fma(r42, r36, r45);
    r10 = r1 * r7;
    r10 = r10 * r60;
    r74 = fma(r32, r10, r74);
    r36 = fma(r5, r36, r41);
    r10 = r47 * r7;
    r10 = r10 * r60;
    r36 = fma(r32, r10, r36);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             2 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r74,
                                             r36);
    r10 = fma(r0, r60, r45);
    r53 = r0 * r13;
    r53 = fma(r42, r53, r16);
    r16 = r53 * r71;
    r16 = r4 <= r51 ? r53 : r16;
    r72 = r53 * r73;
    r16 = r48 < r52 ? r72 : r16;
    r72 = r53 * r56;
    r16 = r48 < r54 ? r72 : r16;
    r16 = r48 < r50 ? r53 : r16;
    r72 = r50 * r16;
    r53 = fma(r53, r57, r58 * r72);
    r53 = r50 * r53;
    r53 = r53 * r59;
    r53 = r4 <= r31 ? r44 : r53;
    r10 = fma(r42, r53, r10);
    r72 = fma(r46, r60, r41);
    r72 = fma(r5, r53, r72);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             4 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r10,
                                             r72);
    r53 = r1 * r13;
    r53 = fma(r47, r3, r42 * r53);
    r43 = r53 * r71;
    r43 = r4 <= r51 ? r53 : r43;
    r75 = r53 * r73;
    r43 = r48 < r52 ? r75 : r43;
    r75 = r53 * r56;
    r43 = r48 < r54 ? r75 : r43;
    r43 = r48 < r50 ? r53 : r43;
    r75 = r50 * r43;
    r75 = fma(r58, r75, r53 * r57);
    r75 = r50 * r75;
    r75 = r75 * r59;
    r75 = r4 <= r31 ? r44 : r75;
    r53 = fma(r42, r75, r45);
    r53 = fma(r1, r60, r53);
    r75 = fma(r5, r75, r41);
    r75 = fma(r47, r60, r75);
    WriteIdx2<1024, double, double, double2>(out_calib_jac,
                                             6 * out_calib_jac_num_alloc,
                                             global_thread_idx,
                                             r53,
                                             r75);
    r76 = r5 * r26;
    r76 = fma(r63, r76, r66 * r37);
    r77 = r5 * r36;
    r77 = fma(r63, r77, r74 * r37);
    WriteSum2<double, double>((double*)inout_shared, r76, r77);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r5 * r72;
    r77 = fma(r10, r37, r63 * r77);
    r76 = r5 * r75;
    r76 = fma(r63, r76, r53 * r37);
    WriteSum2<double, double>((double*)inout_shared, r77, r76);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r66, r66, r26 * r26);
    r77 = fma(r74, r74, r36 * r36);
    WriteSum2<double, double>((double*)inout_shared, r76, r77);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            0 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fma(r10, r10, r72 * r72);
    r76 = fma(r75, r75, r53 * r53);
    WriteSum2<double, double>((double*)inout_shared, r77, r76);
  };
  FlushSumShared<2, double>(out_calib_precond_diag,
                            2 * out_calib_precond_diag_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fma(r26, r36, r66 * r74);
    r77 = fma(r26, r72, r66 * r10);
    WriteSum2<double, double>((double*)inout_shared, r76, r77);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            0 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fma(r66, r53, r26 * r75);
    r77 = fma(r74, r10, r36 * r72);
    WriteSum2<double, double>((double*)inout_shared, r66, r77);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            2 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fma(r74, r53, r36 * r75);
    r53 = fma(r10, r53, r72 * r75);
    WriteSum2<double, double>((double*)inout_shared, r74, r53);
  };
  FlushSumShared<2, double>(out_calib_precond_tril,
                            4 * out_calib_precond_tril_num_alloc,
                            calib_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fma(r21, r29, r28 * r40);
    r53 = r30 * r12;
    r74 = r21 * r6;
    r74 = r74 * r39;
    r74 = fma(r38, r74, r32 * r53);
    r53 = fma(r1, r74, r0 * r28);
    r10 = fma(r60, r53, r45);
    r77 = r13 * r42;
    r74 = fma(r47, r74, r46 * r28);
    r77 = fma(r74, r3, r53 * r77);
    r53 = r77 * r71;
    r53 = r4 <= r51 ? r77 : r53;
    r28 = r77 * r73;
    r53 = r48 < r52 ? r28 : r53;
    r28 = r77 * r56;
    r53 = r48 < r54 ? r28 : r53;
    r53 = r48 < r50 ? r77 : r53;
    r28 = r50 * r53;
    r77 = fma(r77, r57, r58 * r28);
    r77 = r50 * r77;
    r77 = r77 * r59;
    r77 = r4 <= r31 ? r44 : r77;
    r10 = fma(r42, r77, r10);
    r74 = fma(r60, r74, r41);
    r74 = fma(r5, r77, r74);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             0 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r10,
                                             r74);
    r77 = r30 * r24;
    r28 = r35 * r6;
    r28 = r28 * r39;
    r28 = fma(r38, r28, r32 * r77);
    r19 = fma(r35, r29, r19 * r40);
    r77 = fma(r0, r19, r1 * r28);
    r66 = fma(r60, r77, r45);
    r76 = r13 * r42;
    r19 = fma(r46, r19, r47 * r28);
    r76 = fma(r19, r3, r77 * r76);
    r77 = r76 * r71;
    r77 = r4 <= r51 ? r76 : r77;
    r28 = r76 * r73;
    r77 = r48 < r52 ? r28 : r77;
    r28 = r76 * r56;
    r77 = r48 < r54 ? r28 : r77;
    r77 = r48 < r50 ? r76 : r77;
    r28 = r50 * r77;
    r76 = fma(r76, r57, r58 * r28);
    r76 = r50 * r76;
    r76 = r76 * r59;
    r76 = r4 <= r31 ? r44 : r76;
    r66 = fma(r42, r76, r66);
    r76 = fma(r5, r76, r41);
    r76 = fma(r60, r19, r76);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             2 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r66,
                                             r76);
    r19 = r13 * r42;
    r28 = r30 * r33;
    r78 = r27 * r6;
    r78 = r78 * r39;
    r78 = fma(r38, r78, r32 * r28);
    r40 = fma(r23, r40, r27 * r29);
    r23 = fma(r0, r40, r1 * r78);
    r40 = fma(r46, r40, r47 * r78);
    r3 = fma(r40, r3, r23 * r19);
    r71 = r3 * r71;
    r71 = r4 <= r51 ? r3 : r71;
    r73 = r3 * r73;
    r71 = r48 < r52 ? r73 : r71;
    r56 = r3 * r56;
    r71 = r48 < r54 ? r56 : r71;
    r71 = r48 < r50 ? r3 : r71;
    r48 = r50 * r71;
    r57 = fma(r3, r57, r58 * r48);
    r57 = r50 * r57;
    r57 = r57 * r59;
    r57 = r4 <= r31 ? r44 : r57;
    r45 = fma(r42, r57, r45);
    r45 = fma(r60, r23, r45);
    r57 = fma(r5, r57, r41);
    r57 = fma(r60, r40, r57);
    WriteIdx2<1024, double, double, double2>(out_point_jac,
                                             4 * out_point_jac_num_alloc,
                                             global_thread_idx,
                                             r45,
                                             r57);
    r40 = r5 * r74;
    r40 = fma(r63, r40, r10 * r37);
    r41 = r5 * r76;
    r41 = fma(r63, r41, r66 * r37);
    WriteSum2<double, double>((double*)inout_shared, r40, r41);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = r5 * r57;
    r41 = fma(r63, r41, r45 * r37);
    WriteSum1<double, double>((double*)inout_shared, r41);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fma(r10, r10, r74 * r74);
    r37 = fma(r76, r76, r66 * r66);
    WriteSum2<double, double>((double*)inout_shared, r41, r37);
  };
  FlushSumShared<2, double>(out_point_precond_diag,
                            0 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r45, r45, r57 * r57);
    WriteSum1<double, double>((double*)inout_shared, r37);
  };
  FlushSumShared<1, double>(out_point_precond_diag,
                            2 * out_point_precond_diag_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fma(r10, r66, r74 * r76);
    r10 = fma(r74, r57, r10 * r45);
    WriteSum2<double, double>((double*)inout_shared, r37, r10);
  };
  FlushSumShared<2, double>(out_point_precond_tril,
                            0 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = fma(r66, r45, r76 * r57);
    WriteSum1<double, double>((double*)inout_shared, r45);
  };
  FlushSumShared<1, double>(out_point_precond_tril,
                            2 * out_point_precond_tril_num_alloc,
                            point_indices_loc,
                            (double*)inout_shared);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

void PinholeResJacFirst(double* pose,
                        unsigned int pose_num_alloc,
                        SharedIndex* pose_indices,
                        double* calib,
                        unsigned int calib_num_alloc,
                        SharedIndex* calib_indices,
                        double* point,
                        unsigned int point_num_alloc,
                        SharedIndex* point_indices,
                        double* pixel,
                        unsigned int pixel_num_alloc,
                        double* weight_loss,
                        unsigned int weight_loss_num_alloc,
                        double* out_res,
                        unsigned int out_res_num_alloc,
                        double* const out_rTr,
                        double* out_pose_jac,
                        unsigned int out_pose_jac_num_alloc,
                        double* const out_pose_njtr,
                        unsigned int out_pose_njtr_num_alloc,
                        double* const out_pose_precond_diag,
                        unsigned int out_pose_precond_diag_num_alloc,
                        double* const out_pose_precond_tril,
                        unsigned int out_pose_precond_tril_num_alloc,
                        double* out_calib_jac,
                        unsigned int out_calib_jac_num_alloc,
                        double* const out_calib_njtr,
                        unsigned int out_calib_njtr_num_alloc,
                        double* const out_calib_precond_diag,
                        unsigned int out_calib_precond_diag_num_alloc,
                        double* const out_calib_precond_tril,
                        unsigned int out_calib_precond_tril_num_alloc,
                        double* out_point_jac,
                        unsigned int out_point_jac_num_alloc,
                        double* const out_point_njtr,
                        unsigned int out_point_njtr_num_alloc,
                        double* const out_point_precond_diag,
                        unsigned int out_point_precond_diag_num_alloc,
                        double* const out_point_precond_tril,
                        unsigned int out_point_precond_tril_num_alloc,
                        size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeResJacFirstKernel<<<n_blocks, 1024>>>(pose,
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
                                               out_rTr,
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