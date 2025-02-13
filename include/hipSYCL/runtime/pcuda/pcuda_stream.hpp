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

#ifndef ACPP_RT_PCUDA_STREAM_HPP
#define ACPP_RT_PCUDA_STREAM_HPP

#include <memory>

#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"


namespace hipsycl::rt::pcuda {

class pcuda_runtime;
using internal_stream_t = std::shared_ptr<inorder_queue>;

pcudaError_t create_stream(internal_stream_t *&out, pcuda_runtime *,
                           unsigned int flags, int priority);
pcudaError_t destroy_stream(internal_stream_t *stream, pcuda_runtime *);

}

#endif
