/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef HIPSYCL_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../generic/hiplike/warp_shuffle.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA

namespace hipsycl {
namespace sycl::detail::hiplike_builtins {

// barrier
template<int Dim>
__device__
inline void __hipsycl_group_barrier(group<Dim> g, memory_scope fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence();
  }
  __syncthreads();
}

template<int Dim>
__device__
inline void __hipsycl_group_barrier(group<Dim> g) {
  __syncthreads();
}

__device__
inline void __hipsycl_group_barrier(sub_group g, memory_scope fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence();
  } else if (fence_scope == memory_scope::work_group) {
    __threadfence_block();
  }
  __syncwarp(); // not necessarily needed, but might improve performance
}

__device__
inline void __hipsycl_group_barrier(sub_group g) {
  __syncwarp(); // not necessarily needed, but might improve performance
}

template<typename Group, std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
__device__
inline void __hipsycl_group_barrier(Group g, memory_scope fence_scope) {
  if constexpr (Group::fence_scope == memory_scope::sub_group) {
    if (fence_scope == memory_scope::device) {
      __threadfence();
    }
    __syncthreads();
  } else {
    __syncwarp(); // not necessarily needed, but might improve performance
  }
}

template<typename Group, std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
__device__
inline void __hipsycl_group_barrier(Group g) {
  if constexpr (Group::fence_scope == memory_scope::sub_group) {
    __syncthreads();
  } else {
    __syncwarp(); // not necessarily needed, but might improve performance
  }
}

namespace detail {
// any_of
template<typename Group>
__device__
inline bool __hipsycl_any_of_sub_group(Group g, bool pred) {
  return __any_sync(detail::AllMask, pred);
}

// all_of
template<typename Group>
__device__
inline bool __hipsycl_all_of_sub_group(Group g, bool pred) {
  return __all_sync(detail::AllMask, pred);
}

// none_of
template<typename Group>
__device__
inline bool __hipsycl_none_of_sub_group(Group g, bool pred) {
  return !__any_sync(detail::AllMask, pred);
}

// reduce
template<typename T, typename BinaryOperation>
__device__
T __hipsycl_reduce_over_sub_group(sub_group g, T x, BinaryOperation binary_op) {
  const size_t lid = g.get_local_linear_id();
  const size_t lrange = g.get_local_linear_range();
  const unsigned int activemask = __ballot_sync(0xffffffff, 1);

  auto local_x = x;

  for (size_t i = lrange / 2; i > 0; i /= 2) {
    auto other_x = detail::__hipsycl_shuffle_impl(local_x, lid + i);
    if (activemask & (1 << (lid + i)))
      local_x = binary_op(local_x, other_x);
  }
  return detail::__hipsycl_shuffle_impl(local_x, 0);
}

} //namespace detail

// exclusive_scan
template<typename T, typename BinaryOperation>
__device__
T __hipsycl_inclusive_scan_over_group(sub_group g, T x, BinaryOperation binary_op) {
  auto local_x = x;
  const size_t lid = g.get_local_linear_id();

  const unsigned int activemask = __ballot_sync(0xffffffff, 1);

  const size_t lrange = g.get_local_linear_range();


  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::__hipsycl_shuffle_impl(local_x, next_id);
    if (activemask & (1 << (next_id)) && i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  return local_x;
}

// exclusive_scan
template<typename V, typename T, typename BinaryOperation>
__device__
T __hipsycl_exclusive_scan_over_group(sub_group g, V x, T init, BinaryOperation binary_op) {
  const size_t lid = g.get_local_linear_id();
  auto local_x = x;

  local_x = detail::__hipsycl_shuffle_up_impl(local_x, 1);
  if (lid == 0)
    local_x = init;

  return __hipsycl_inclusive_scan_over_group(g, local_x, binary_op);
}

} // namespace sycl::detail::hiplike_builtins
} // namespace hipsycl

#endif
#endif // HIPSYCL_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP
