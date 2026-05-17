#pragma once

#include <cuda_runtime.h>

namespace caspar {

cudaError_t ConstDepthScaleStackedToCaspar(const float* stacked_data,
                                           float* cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t ConstDepthScaleCasparToStacked(const float* cas_data,
                                           float* stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t ConstInvStd1StackedToCaspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstInvStd1CasparToStacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstInvStd4StackedToCaspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstInvStd4CasparToStacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t ConstLogDepthStackedToCaspar(const float* stacked_data,
                                         float* cas_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstLogDepthCasparToStacked(const float* cas_data,
                                         float* stacked_data,
                                         const unsigned int cas_stride,
                                         const unsigned int cas_offset,
                                         const unsigned int num_objects);

cudaError_t ConstPinholeCalibStackedToCaspar(const float* stacked_data,
                                             float* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeCalibCasparToStacked(const float* cas_data,
                                             float* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeFocalStackedToCaspar(const float* stacked_data,
                                             float* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholeFocalCasparToStacked(const float* cas_data,
                                             float* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t ConstPinholePoseStackedToCaspar(const float* stacked_data,
                                            float* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePoseCasparToStacked(const float* cas_data,
                                            float* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholePrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstPinholeRotationStackedToCaspar(const float* stacked_data,
                                                float* cas_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstPinholeRotationCasparToStacked(const float* cas_data,
                                                float* stacked_data,
                                                const unsigned int cas_stride,
                                                const unsigned int cas_offset,
                                                const unsigned int num_objects);

cudaError_t ConstPixelStackedToCaspar(const float* stacked_data,
                                      float* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPixelCasparToStacked(const float* cas_data,
                                      float* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointStackedToCaspar(const float* stacked_data,
                                      float* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstPointCasparToStacked(const float* cas_data,
                                      float* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t ConstReprojectionWeightLossStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstReprojectionWeightLossCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstRobustLossStackedToCaspar(const float* stacked_data,
                                           float* cas_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t ConstRobustLossCasparToStacked(const float* cas_data,
                                           float* stacked_data,
                                           const unsigned int cas_stride,
                                           const unsigned int cas_offset,
                                           const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndDistortionStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialFocalAndDistortionCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPoseCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t ConstSimpleRadialPrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t DepthScaleStackedToCaspar(const float* stacked_data,
                                      float* cas_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t DepthScaleCasparToStacked(const float* cas_data,
                                      float* stacked_data,
                                      const unsigned int cas_stride,
                                      const unsigned int cas_offset,
                                      const unsigned int num_objects);

cudaError_t PinholeCalibStackedToCaspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeCalibCasparToStacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalStackedToCaspar(const float* stacked_data,
                                        float* cas_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholeFocalCasparToStacked(const float* cas_data,
                                        float* stacked_data,
                                        const unsigned int cas_stride,
                                        const unsigned int cas_offset,
                                        const unsigned int num_objects);

cudaError_t PinholePoseStackedToCaspar(const float* stacked_data,
                                       float* cas_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePoseCasparToStacked(const float* cas_data,
                                       float* stacked_data,
                                       const unsigned int cas_stride,
                                       const unsigned int cas_offset,
                                       const unsigned int num_objects);

cudaError_t PinholePrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholePrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t PinholeTranslationStackedToCaspar(const float* stacked_data,
                                              float* cas_data,
                                              const unsigned int cas_stride,
                                              const unsigned int cas_offset,
                                              const unsigned int num_objects);

cudaError_t PinholeTranslationCasparToStacked(const float* cas_data,
                                              float* stacked_data,
                                              const unsigned int cas_stride,
                                              const unsigned int cas_offset,
                                              const unsigned int num_objects);

cudaError_t PointStackedToCaspar(const float* stacked_data,
                                 float* cas_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t PointCasparToStacked(const float* cas_data,
                                 float* stacked_data,
                                 const unsigned int cas_stride,
                                 const unsigned int cas_offset,
                                 const unsigned int num_objects);

cudaError_t SimpleRadialCalibStackedToCaspar(const float* stacked_data,
                                             float* cas_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialCalibCasparToStacked(const float* cas_data,
                                             float* stacked_data,
                                             const unsigned int cas_stride,
                                             const unsigned int cas_offset,
                                             const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndDistortionStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialFocalAndDistortionCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPoseStackedToCaspar(const float* stacked_data,
                                            float* cas_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPoseCasparToStacked(const float* cas_data,
                                            float* stacked_data,
                                            const unsigned int cas_stride,
                                            const unsigned int cas_offset,
                                            const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointStackedToCaspar(
    const float* stacked_data,
    float* cas_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

cudaError_t SimpleRadialPrincipalPointCasparToStacked(
    const float* cas_data,
    float* stacked_data,
    const unsigned int cas_stride,
    const unsigned int cas_offset,
    const unsigned int num_objects);

}  // namespace caspar