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



#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"


namespace hipsycl::rt::pcuda {

inorder_queue* get_queue(pcudaStream_t stream) {
  return static_cast<internal_stream_t*>(stream)->get();
}


class thread_local_state {
private:
  int _current_device;
  int _current_platform;
  int _current_backend;
};


ACPP_PCUDA_API void __pcudaPushCallConfiguration(dim3 grid, dim3 block,
                                                 size_t shared_mem = 0,
                                                 pcudaStream_t stream = nullptr) {

}

ACPP_PCUDA_API pcudaError_t __pcudaKernelCall(const char *kernel_name,
                                              void **args,
                                              std::size_t hcf_object) {

}

ACPP_PCUDA_API pcudaError_t pcudaGetLastError();

}