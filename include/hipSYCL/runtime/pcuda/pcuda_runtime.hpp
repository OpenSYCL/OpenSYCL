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

#ifndef ACPP_RT_PCUDA_RUNTIME_HPP
#define ACPP_RT_PCUDA_RUNTIME_HPP

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_device_topology.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"

namespace hipsycl::rt {

class runtime;

}

namespace hipsycl::rt::pcuda {

class pcuda_runtime {
public:
  ~pcuda_runtime();

  runtime* get_rt() const {
    return _rt.get();
  }

  const device_topology& get_topology() const {
    return _topology;
  }

  const thread_local_state& get_tls_state() const;
private:
  runtime_keep_alive_token _rt;
  device_topology _topology;
};

}

#endif
