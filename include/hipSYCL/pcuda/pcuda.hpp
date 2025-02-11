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
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include <cstddef>

#include "pcuda_runtime.hpp"

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

#define __global__                                                             \
  [[clang::annotate("hipsycl_sscp_kernel")]]                                   \
  [[clang::annotate("hipsycl_sscp_outlining")]]                                \
  [[clang::annotate("acpp_free_kernel")]]                                      \
  [[clang::annotate("hipsycl_kernel_dimension", 3)]]





// needs -fdeclspec
struct __pcudaThreadIdx {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return __acpp_sscp_get_local_id_x();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return __acpp_sscp_get_local_id_y();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return __acpp_sscp_get_local_id_z();
  }
};

struct __pcudaBlockIdx {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return __acpp_sscp_get_group_id_x();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return __acpp_sscp_get_group_id_y();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return __acpp_sscp_get_group_id_z();
  }
};

struct __pcudaBlockDim {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return __acpp_sscp_get_local_size_x();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return __acpp_sscp_get_local_size_y();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return __acpp_sscp_get_local_size_z();
  }
};

struct __pcudaGridDim {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return __acpp_sscp_get_local_size_x();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return __acpp_sscp_get_local_size_y();
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return __acpp_sscp_get_local_size_z();
  }
};

template<class F>
[[clang::annotate("hipsycl_sscp_kernel")]]
[[clang::annotate("hipsycl_sscp_outlining")]]
[[clang::annotate("hipsycl_kernel_dimension", 3)]]
void __pcuda_kernel(const F& f){
  if (__acpp_sscp_is_device) {
    F g = f;
    g();
  }
}

namespace __sscp_dispatch {

template<class F>
class pcuda_wrapper {
public:
  pcuda_wrapper(F f)
  : _f {f} {}

  void operator()() {
    _f();
  }
private:
  F _f;
};

}

extern const __pcudaThreadIdx threadIdx;
extern const __pcudaBlockIdx blockIdx;
extern const __pcudaBlockDim blockDim;
extern const __pcudaGridDim gridDim;

#define warpSize __acpp_sscp_get_subgroup_max_size()

template <class F>
inline pcudaError_t pcudaSubmit(dim3 grid, dim3 block, size_t shared_mem,
                                pcudaStream_t stream, F f) {
  __pcudaPushCallConfiguration(grid, block, shared_mem, stream);
  __pcuda_kernel(__sscp_dispatch::pcuda_wrapper{f});
  return pcudaGetLastError();
}

template<class F>
inline pcudaError_t pcudaSubmit(dim3 grid, dim3 block, size_t shared_mem, F f) {
  return pcudaSubmit(grid, block, 0, nullptr, f);
}

template<class F>
inline pcudaError_t pcudaSubmit(dim3 grid, dim3 block, F f) {
  return pcudaSubmit(grid, block, 0, f);
}

#define PCUDA_KERNEL_NAME(...) __VA_ARGS__
#define PCUDA_SYMBOL(X) X
#define pcudaLaunchKernelGGL(kernel_name, grid, block, shared_mem, stream, ...) \
  pcudaSubmit(grid, block, shared_mem, stream, [=](){ kernel_name(__VA_ARGS__); })

#endif
