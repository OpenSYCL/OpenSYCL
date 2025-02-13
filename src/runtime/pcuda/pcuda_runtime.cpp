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

#include <memory>
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"



namespace hipsycl::rt::pcuda {

namespace {
thread_local std::unique_ptr<thread_local_state> tls_state;
}


const thread_local_state& pcuda_runtime::get_tls_state() const {
  if(!tls_state)
    tls_state.reset(new thread_local_state{this});
  return *tls_state.get();
}

}


