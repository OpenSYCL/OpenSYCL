/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay and contributors
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

#ifndef HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP
#define HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/glue/generic/code_object.hpp"
#include "hipSYCL/glue/llvm-sscp/s1_ir_constants.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"
#include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "ir_constants.hpp"

#include <array>
#include <string_view>


template <typename KernelType>
// hipsycl_sscp_kernel causes kernel entries to be emitted to the HCF
[[clang::annotate("hipsycl_sscp_kernel")]]
// hipsycl_sscp_outlining creates an entrypoint for outlining of device code
[[clang::annotate("hipsycl_sscp_outlining")]]
void __acpp_sscp_kernel(const KernelType& kernel) {
  if(__acpp_sscp_is_device) {
    // The copy here creates an alloca that can help inferring the argument
    // type in case of opaque pointers.
    KernelType k = kernel;
    k();
  }
}


// hipSYCL SSCP LLVM magic will add definition, but clang warns - suppress until
// we find a better solution to implement things
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-internal" 
// The SSCP compiler will use this invocation to connect the kernel function
// to a global variable where the kernel name can be stored.
// First argument has to be a function pointer to the kernel,
// second one a pointer to a global variable.
// No indirection is allowed! If I say, the argument has to be a global variable,
// I mean it. Directly. No passing through other functions first.
template <class Kernel>
void __acpp_sscp_extract_kernel_name(void (*Func)(const Kernel&),
                                        const char *target);
#pragma clang diagnostic pop

namespace hipsycl {
namespace glue {

namespace sscp {

static std::string get_local_hcf_object() {
  return std::string{
      reinterpret_cast<const char *>(__acpp_local_sscp_hcf_content),
      __acpp_local_sscp_hcf_object_size};
}

// TODO: Maybe this can be unified with the HIPSYCL_STATIC_HCF_REGISTRATION
// macro. We cannot use this macro directly because it expects
// the object id to be constexpr, which it is not for the SSCP case.
struct static_hcf_registration {
  static_hcf_registration(const std::string& hcf_data) {
    this->_hcf_object = rt::hcf_cache::get().register_hcf_object(
        common::hcf_container{hcf_data});
  }

  ~static_hcf_registration() {
    rt::hcf_cache::get().unregister_hcf_object(_hcf_object);
  }
private:
  rt::hcf_object_id _hcf_object;
};
static static_hcf_registration
    __acpp_register_sscp_hcf_object{get_local_hcf_object()};


}

// Do not change this namespace name - compiler may look for
// this name to identify structs passed in as kernel arguments.
namespace __sscp_dispatch {

template <int Dimensions, bool WithOffset>
bool item_is_in_range(const sycl::item<Dimensions, WithOffset> &item,
                      const sycl::range<Dimensions> &execution_range,
                      const sycl::id<Dimensions>& offset = {}) {

  for(int i = 0; i < Dimensions; ++i) {
    if constexpr(WithOffset) {
      if(item.get_id(i) >= offset.get(i) + execution_range.get(i))
        return false;
    } else {
      if(item.get_id(i) >= execution_range.get(i))
        return false;
    }
  }
  return true;
}

template<class UserKernel>
class single_task {
public:
  single_task(const UserKernel& k)
  : _k{k} {}

  [[clang::annotate("hipsycl_kernel_dimension", 0)]]
  void operator()() const {
    _k();
  }
private:
  UserKernel _k;
};

template<class UserKernel, int Dimensions>
class basic_parallel_for {
public:
  basic_parallel_for(const UserKernel &k,
                     sycl::range<Dimensions> execution_range)
      : _k{k}, _range{execution_range} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    auto this_item = sycl::detail::make_item<Dimensions>(
      sycl::detail::get_global_id<Dimensions>(), _range
    );
    if(item_is_in_range(this_item, _range))
      _k(this_item);
  }
private:
  UserKernel _k;
  sycl::range<Dimensions> _range;
};

template<class UserKernel, int Dimensions>
class basic_parallel_for_offset {
public:
  basic_parallel_for_offset(const UserKernel &k, sycl::id<Dimensions> offset,
                            sycl::range<Dimensions> execution_range)
      : _k{k}, _range{execution_range}, _offset{offset} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    auto this_item = sycl::detail::make_item<Dimensions>(
        sycl::detail::get_global_id<Dimensions>() + _offset, _range, _offset);
    
    if(item_is_in_range(this_item, _range, _offset))
      _k(this_item);
  }
private:
  UserKernel _k;
  sycl::range<Dimensions> _range;
  sycl::id<Dimensions> _offset;
};


template<class UserKernel, int Dimensions>
class ndrange_parallel_for {
public:
  ndrange_parallel_for(const UserKernel& k)
  : _k{k} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    const sycl::id<Dimensions> zero_offset{};
    sycl::nd_item<Dimensions> this_item{
        &zero_offset, sycl::detail::get_group_id<Dimensions>(),
        sycl::detail::get_local_id<Dimensions>(),
        sycl::detail::get_local_size<Dimensions>(),
        sycl::detail::get_grid_size<Dimensions>()};

    _k(this_item);
  };
private:
  UserKernel _k;
};

template<class UserKernel, int Dimensions>
class ndrange_parallel_for_offset {
public:
  ndrange_parallel_for_offset(const UserKernel& k, sycl::id<Dimensions> offset)
  : _k{k}, _offset{offset} {}

  [[clang::annotate("hipsycl_kernel_dimension", Dimensions)]]
  void operator()() const {
    sycl::nd_item<Dimensions> this_item{
        &_offset, sycl::detail::get_group_id<Dimensions>(),
        sycl::detail::get_local_id<Dimensions>(),
        sycl::detail::get_local_size<Dimensions>(),
        sycl::detail::get_grid_size<Dimensions>()};

    _k(this_item);
  };
private:
  UserKernel _k;
  const sycl::id<Dimensions> _offset;
};

}

class sscp_kernel_launcher : public rt::backend_kernel_launcher
{
public:

  sscp_kernel_launcher() {}
  virtual ~sscp_kernel_launcher(){}

  virtual void set_params(void* params) override {
    _params = params;
  }

  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k) {

    this->_type = type;
    this->_hcf_object_id = __acpp_local_sscp_hcf_object_id;

    if constexpr(type == rt::kernel_type::single_task){

      configure_launch_with_global_range(__sscp_dispatch::single_task{k},
                                         sycl::range{1}, sycl::range{1},
                                         dynamic_local_memory);

    } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

      if(offset == sycl::id<Dim>{}) {
        configure_launch_with_global_range(
            __sscp_dispatch::basic_parallel_for{k, global_range}, global_range,
            local_range, dynamic_local_memory);
      } else {
        configure_launch_with_global_range(
            __sscp_dispatch::basic_parallel_for_offset{k, offset, global_range},
            global_range, local_range, dynamic_local_memory);
      }

    } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

      if(offset == sycl::id<Dim>{}) {
        configure_launch_with_global_range(
            __sscp_dispatch::ndrange_parallel_for<Kernel, Dim>{k}, global_range,
            local_range, dynamic_local_memory);
      } else {
        configure_launch_with_global_range(
            __sscp_dispatch::ndrange_parallel_for_offset<Kernel, Dim>{k, offset},
            global_range, local_range, dynamic_local_memory);
      }

    } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

    } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {
      
    } else if constexpr (type == rt::kernel_type::custom) {
      // handled at invoke time
      _custom_op = k;
    }
    else {
      assert(false && "Unsupported kernel type");
    }
  }

  virtual int get_backend_score(rt::backend_id b) const final override {
    // The other backends return 2 for exact matches,
    // so this means that SSCP is currently preferred when no
    // other exactly matching backend kernel launcher was found.
    // TODO: Should we prevent selection of SSCP if the backend
    // does not support SSCP runtime compilation?
    return 1;
  }

  virtual void invoke(rt::dag_node *node,
                      const rt::kernel_configuration &config) final override {
    assert(node);

    if(_type == rt::kernel_type::custom) {
      assert(_params);
      sycl::interop_handle handle{node->get_assigned_device(),
                                  _params};

      _custom_op(handle);
    } else {
      auto *kernel_op =
          static_cast<rt::kernel_operation *>(node->get_operation());

      kernel_op->initialize_embedded_pointers(
          static_cast<void *>(_kernel_args.data()), _kernel_args.size());

      auto sscp_invoker = this->get_launch_capabilities().get_sscp_invoker();
      if(!sscp_invoker) {
        rt::register_error(
          __acpp_here(),
          rt::error_info{"Attempted to prepare to launch SSCP kernel, but the backend "
                         "did not configure the kernel launcher for SSCP."});
      }
      auto *invoker = sscp_invoker.value();

      auto selected_group_size = _group_size;
      if (_group_size.size() == 0)
        selected_group_size =
            invoker->select_group_size(_global_size, _group_size);
      
      rt::range<3> num_groups;
      for(int i = 0; i < 3; ++i) {
        num_groups[i] = (_global_size[i] + selected_group_size[i] - 1) /
                        selected_group_size[i];
      }

      std::array<const void*, 1> args{_kernel_args.data()};
      std::size_t arg_size = _kernel_args.size();

      auto err = invoker->submit_kernel(
          *kernel_op, _hcf_object_id, num_groups, selected_group_size,
          _local_mem_size, const_cast<void **>(args.data()), &arg_size,
          args.size(), _kernel_name, config);

      if(!err.is_success()) {
        rt::register_error(err);
      }
    }
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:
  template <int Dim>
  rt::range<3> flip_range(const sycl::range<Dim> &r) {
    rt::range<3> rt_range{1,1,1};

    for (int i = 0; i < Dim; ++i) {
      rt_range[i] = r[Dim - i - 1];
    }

    return rt_range;
  }

  template <class Kernel, int Dim>
  void configure_launch_with_global_range(const Kernel &k,
                                          const sycl::range<Dim> &global_range,
                                          const sycl::range<Dim> &group_size,
                                          unsigned local_mem_size) {

    _global_size = flip_range(global_range);
    _group_size = flip_range(group_size);
    _local_mem_size = local_mem_size;
    _kernel_args.resize(sizeof(Kernel));
    std::memcpy(_kernel_args.data(), &k, sizeof(Kernel));

    _kernel_name = generate_kernel(k);
  }


  // Generate SSCP kernel and return name of the generated kernel
  template<class Kernel>
  static std::string_view generate_kernel(const Kernel& k) {
    if (__acpp_sscp_is_device) {
      __acpp_sscp_kernel(k);
    }

    // Compiler will change the number of elements to the kernel name length
    static char __acpp_sscp_kernel_name [] = "kernel-name-extraction-failed";

    __acpp_sscp_extract_kernel_name<Kernel>(
        &__acpp_sscp_kernel<Kernel>,
        &__acpp_sscp_kernel_name[0]);
    return &__acpp_sscp_kernel_name[0];
  }
  
  void* _params = nullptr;

  // the type of the kernel launch
  rt::kernel_type _type;
  std::vector<uint8_t> _kernel_args;
  rt::range<3> _global_size;
  rt::range<3> _group_size;
  unsigned _local_mem_size;
  std::string_view _kernel_name;
  unsigned long long _hcf_object_id;

  std::function<void(sycl::interop_handle&)> _custom_op;
};


}
}

#endif
