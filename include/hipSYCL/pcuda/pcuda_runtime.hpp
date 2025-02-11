
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

#ifndef ACPP_PCUDA_RUNTIME_HPP
#define ACPP_PCUDA_RUNTIME_HPP

#include "detail/dim3.hpp"
#include <cstdlib>

#define ACPP_PCUDA_API extern "C"



typedef enum pcudaError : int {
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
ACPP_PCUDA_API pcudaError_t __pcudaKernelCall(const char *kernel_name,
                                              void **args,
                                              std::size_t hcf_object);

ACPP_PCUDA_API pcudaError_t pcudaLaunchKernel(const void *func, dim3 grid,
                                              dim3 block, void **args,
                                              size_t shared_mem,
                                              pcudaStream_t stream);

ACPP_PCUDA_API pcudaError_t pcudaGetLastError();



#endif
