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
#ifndef ACPP_OCL_QUERY
#define ACPP_OCL_QUERY
#include "hipSYCL/runtime/error.hpp"

#include <CL/opencl.h>
#include <CL/opencl.hpp>

namespace hipsycl::rt {

template <cl_device_info Query, class ResultT>
ResultT info_query(const cl::Device &dev) {
  ResultT r{};
  cl_int err = dev.getInfo(Query, &r);
  if (err != CL_SUCCESS) {
    register_error(
        __acpp_here(),
        error_info{"ocl: Could not obtain device info", error_code{"CL", err}});
  }

  return r;
}

} // namespace hipsycl::rt

#endif