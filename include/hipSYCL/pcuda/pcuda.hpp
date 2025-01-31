/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause


#ifndef ACPP_PCUDA_HPP
#define ACPP_PCUDA_HPP

#include "hipSYCL/glue/llvm-sscp/s1_ir_constants.hpp"
#include "hipSYCL/glue/llvm-sscp/hcf_registration.hpp"
#include "hipSYCL/sycl/libkernel/sscp/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/subgroup.hpp"
#include <cstddef>

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#define __global__                                                             \
  [[clang::annotate("hipsycl_sscp_kernel")]]                                   \
  [[clang::annotate("hipsycl_sscp_outlining")]]                                \
  [[clang::annotate("acpp_free_kernel")]]

#define ACPP_PCUDA_API extern "C"

struct dim3 {
  dim3(unsigned x_=1, unsigned y_=1, unsigned z_=1)
  : x{x_}, y{y_}, z{z_} {}

  unsigned x;
  unsigned y;
  unsigned z;
};

// needs -fdeclspec
struct __pcudaThreadIdx {
  __declspec(property(get = __acpp_sscp_get_local_id_x)) unsigned x;
  __declspec(property(get = __acpp_sscp_get_local_id_y)) unsigned y;
  __declspec(property(get = __acpp_sscp_get_local_id_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }
};

struct __pcudaBlockIdx {
  __declspec(property(get = __acpp_sscp_get_group_id_x)) unsigned x;
  __declspec(property(get = __acpp_sscp_get_group_id_y)) unsigned y;
  __declspec(property(get = __acpp_sscp_get_group_id_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }
};

struct __pcudaBlockDim {
  __declspec(property(get = __acpp_sscp_get_local_size_x)) unsigned x;
  __declspec(property(get = __acpp_sscp_get_local_size_y)) unsigned y;
  __declspec(property(get = __acpp_sscp_get_local_size_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }
};

struct __pcudaGridDim {
  __declspec(property(get = __acpp_sscp_get_num_groups_x)) unsigned x;
  __declspec(property(get = __acpp_sscp_get_num_groups_y)) unsigned y;
  __declspec(property(get = __acpp_sscp_get_num_groups_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }
};

extern const __pcudaThreadIdx threadIdx;
extern const __pcudaBlockIdx blockIdx;
extern const __pcudaBlockDim blockDim;
extern const __pcudaGridDim gridDim;

#define warpSize __acpp_sscp_get_subgroup_max_size()

typedef enum pcudaError {
  pcudaSuccess,
  pcudaErrorMissingConfiguration,
  pcudaErrorMemoryAllocation,
  pcudaErrorInitializationError,
  pcudaErrorLaunchFailure,
  pcudaErrorPriorLaunchFailure,
  pcudaErrorLaunchTimeout,
  pcudaErrorLaunchOutOfResources,
  pcudaErrorInvalidDeviceFunction,
  pcudaErrorInvalidConfiguration,
  pcudaErrorInvalidDevice,
  pcudaErrorInvalidValue,
  pcudaErrorInvalidPitchValue,
  pcudaErrorInvalidSymbol,
  pcudaErrorMapBufferObjectFailed,
  pcudaErrorUnmapBufferObjectFailed,
  pcudaErrorInvalidHostPointer,
  pcudaErrorInvalidDevicePointer,
  pcudaErrorInvalidTexture,
  pcudaErrorInvalidTextureBinding,
  pcudaErrorInvalidChannelDescriptor,
  pcudaErrorInvalidMemcpyDirection,
  pcudaErrorAddressOfConstant,
  pcudaErrorTextureFetchFailed,
  pcudaErrorTextureNotBound,
  pcudaErrorSynchronizationError,
  pcudaErrorInvalidFilterSetting,
  pcudaErrorInvalidNormSetting,
  pcudaErrorMixedDeviceExecution,
  pcudaErrorCudartUnloading,
  pcudaErrorUnknown,
  pcudaErrorNotYetImplemented,
  pcudaErrorMemoryValueTooLarge,
  pcudaErrorInvalidResourceHandle,
  pcudaErrorNotReady,
  pcudaErrorInsufficientDriver,
  pcudaErrorSetOnActiveProcess,
  pcudaErrorInvalidSurface,
  pcudaErrorNoDevice,
  pcudaErrorECCUncorrectable,
  pcudaErrorSharedObjectSymbolNotFound,
  pcudaErrorSharedObjectInitFailed,
  pcudaErrorUnsupportedLimit,
  pcudaErrorDuplicateVariableName,
  pcudaErrorDuplicateTextureName,
  pcudaErrorDuplicateSurfaceName,
  pcudaErrorDevicesUnavailable,
  pcudaErrorInvalidKernelImage,
  pcudaErrorNoKernelImageForDevice,
  pcudaErrorIncompatibleDriverContext,
  pcudaErrorPeerAccessAlreadyEnabled,
  pcudaErrorPeerAccessNotEnabled,
  pcudaErrorDeviceAlreadyInUse,
  pcudaErrorProfilerDisabled,
  pcudaErrorProfilerNotInitialized,
  pcudaErrorProfilerAlreadyStarted,
  pcudaErrorProfilerAlreadyStopped,
  pcudaErrorStartupFailure,
  pcudaErrorApiFailureBase
} pcudaError_t;

using pcudaStream_t = void*;

ACPP_PCUDA_API void __pcudaPushCallConfiguration(dim3 grid, dim3 block,
                                                 size_t shared_mem = 0,
                                                 pcudaStream_t stream = nullptr);
ACPP_PCUDA_API void __pcudaKernelCall(const char *kernel_name, void **args);

ACPP_PCUDA_API pcudaError_t pcudaLaunchKernel(const void *func, dim3 grid,
                                              dim3 block, void **args,
                                              size_t shared_mem,
                                              pcudaStream_t stream);

#endif
