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
#ifndef HIPSYCL_COLLECTIVE_EXECUTION_ENGINE_HPP
#define HIPSYCL_COLLECTIVE_EXECUTION_ENGINE_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

/**
 * Allow disabling fibers; and don't try using them in device pass.
 */
#if !defined(ACPP_NO_FIBERS) && !defined(SYCL_DEVICE_ONLY)
#define ACPP_HAS_FIBERS
#endif

#ifdef ACPP_HAS_FIBERS

#include <functional>
#include <vector>

#include "tina.h"

#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"

#include "iterate_range.hpp"
#include "range_decomposition.hpp"

namespace hipsycl {
namespace glue {
namespace host {

enum class group_execution_iteration {
  omp_for,
  sequential
};

namespace yield_kind {
// Some odd value not easily confused with pointers
static void* spawn = reinterpret_cast<void*>(0xff07);
static void* barrier = reinterpret_cast<void*>(0xff09);
static void* next_item = reinterpret_cast<void*>(0xff11);
}
static constexpr size_t fiber_stack_size = 256*1024;


template<int Dim>
class collective_execution_engine {
public:
  collective_execution_engine(
      sycl::range<Dim> num_groups, sycl::range<Dim> local_size,
      sycl::id<Dim> offset,
      const static_range_decomposition<Dim>& group_range_decomposition,
      int my_group_region)
      : _num_groups{num_groups}, _local_size{local_size}, _offset{offset},
        _fibers_spawned{false}, _fibers(local_size.size(), nullptr),
        _groups{group_range_decomposition}, _my_group_region{my_group_region},
        _current_coro{nullptr} {}


  template <class WorkItemFunction>
  void run_kernel(WorkItemFunction f) {
    _kernel = f;
    _fibers_spawned = false;
    _master_group_position = 0;

    // Create master fiber
    _fibers[0] = tina_init(nullptr, fiber_stack_size, &master_coro, this);

    bool all_done = false;

    // Launch master fiber
    void* result = resume(_fibers[0]);
    if (_fibers[0]->completed) {
      all_done = true;
    } else {
      // Encountered a barrier, call for help
      assert(result == yield_kind::spawn);
      spawn_fibers();
    }

    while (!all_done) {
      all_done = true;
      // Process all coroutines
      // The only way for the coroutine to yield is either a barrier or quitting.
      // We can't ensure whether all the work-items hit the *same* barrier, and 
      // finished work-iterms don't have to wait on barriers, 
      // so we just cycle through all active coroutines and hope for the best.
      void* master_yield_kind = nullptr;
      for (auto& coro : _fibers) {
        assert(coro);
        if (!coro->completed) {
          void* result = resume(coro);
          if (!coro->completed) {
            assert(result == yield_kind::barrier || result == yield_kind::next_item);
            if (master_yield_kind == nullptr) master_yield_kind = result;
            assert(result == master_yield_kind && "Fibers yielded for different reasons on the same pass");
            all_done = false;
          }
        }
      }
    };

    // Cleanup
    for(auto& coro : _fibers) {
      if(coro) {
        free(coro->buffer); // This frees coro itself too
        coro = nullptr;
      }
    }
  }

  void* resume(tina* coro) {
    _current_coro = coro;  // Track current coroutine for barriers
    void* result = tina_resume(coro, nullptr);
    _current_coro = nullptr;
    return result;
  }

  void barrier() {
    assert(_current_coro != nullptr && "Barrier called outside coroutine context");
    if (!_fibers_spawned) {
      // Technically, we can only yield once here, but it's more explicit this way
      tina_yield(_current_coro, yield_kind::spawn);
      tina_yield(_current_coro, yield_kind::next_item);
    }
    tina_yield(_current_coro, yield_kind::barrier);
  }

private:
  sycl::range<Dim> _num_groups;
  sycl::range<Dim> _local_size;
  sycl::id<Dim> _offset;
  bool _fibers_spawned;
  std::vector<tina*> _fibers;
  std::function<void(sycl::id<Dim>, sycl::id<Dim>)> _kernel;
  size_t _master_group_position;
  const static_range_decomposition<Dim>& _groups;
  int _my_group_region;

  tina* _current_coro;

  static void* master_coro(tina* coro, void* arg) {
    auto* engine = static_cast<collective_execution_engine*>(coro->user_data);
    engine->master_coro_body(coro);
    return nullptr;
  }

  void master_coro_body(tina* coro) {
    _groups.for_each_local_element(
      _my_group_region, [this, coro](sycl::id<Dim> group_id) {
        if(!_fibers_spawned) {
          iterate_range(_local_size, [&](sycl::id<Dim> local_id) {
            if(!_fibers_spawned) {
              execute_work_item(local_id, group_id);
            }
          });
        } else {
          assert(coro == _current_coro);
          tina_yield(coro, yield_kind::next_item);
          assert(coro == _current_coro);
          execute_work_item(sycl::id<Dim>{}, group_id);
        }
        ++_master_group_position;
      });
  }

  struct CoroutineData {
    collective_execution_engine* engine;
    sycl::id<Dim> local_id;
    size_t master_offset;
  };

  static void* worker_coro(tina* coro, void* arg) {
    auto* data = static_cast<CoroutineData*>(coro->user_data);
    auto* engine = data->engine;
    const auto local_id = data->local_id;
    const auto master_offset = data->master_offset;
    delete data;

    engine->worker_coro_body(coro, local_id, master_offset);
    return nullptr;
  }

  void worker_coro_body(tina* coro, sycl::id<Dim> local_id, size_t master_offset) {
    size_t current_group = 0;
    _groups.for_each_local_element(
      _my_group_region, [&](sycl::id<Dim> group_id) {
        if(current_group >= master_offset) {
          assert(coro == _current_coro);
          tina_yield(coro, yield_kind::next_item);
          assert(coro == _current_coro);
          execute_work_item(local_id, group_id);
        }
        current_group++;
      });
  }

  void spawn_fibers() {
    size_t n = 0;
    iterate_range(_local_size, [&](sycl::id<Dim> local_id) {
      if (n != 0) {
        // TODO: Use PMR for allocation of CoroutineData and tina stack
        auto* data = new CoroutineData{this, local_id, _master_group_position};
        _fibers[n] = tina_init(nullptr, fiber_stack_size, &worker_coro, data);
      }
      n++;
    });
    _fibers_spawned = true;
  }

  void execute_work_item(sycl::id<Dim> local_id, sycl::id<Dim> group_id) {
    _kernel(local_id, group_id);
  }

};

}
}
} // namespace hipsycl

#endif // HIPSYCL_HAS_FIBERS

#endif
