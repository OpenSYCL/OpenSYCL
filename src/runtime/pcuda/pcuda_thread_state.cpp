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

#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"

#include <unordered_map>
#include <cstdint>

namespace hipsycl::rt::pcuda {

thread_local_state::thread_local_state(pcuda_runtime* rt)
: _rt{rt}, _current_backend{0}, _current_platform{0}, _current_device{0} {

  auto& topo = rt->get_topology();
  
  int best_backend = 0;
  int best_platform = 0;
  int best_score = -1;

  for(int i = 0; i < topo.all_backends().size(); ++i) {
    backend* b = topo.get_backend(i)->backend_ptr;

    if(b->get_hardware_manager()->get_num_devices() > 0) {
      for(int j = 0; j < topo.get_backend(i)->platforms.size(); ++j) {
        int platform_score = 0;

        for(int k = 0; k < topo.get_platform(i, j)->devices.size(); ++k) {
          auto* dev = topo.get_device(i, j, k)->dev;
          auto dev_id = topo.get_device(i, j, k)->rt_device_id;

          if(dev->is_cpu()) {
            // Prefer OpenCL CPU device over OpenMP one
            // (users can always set ACPP_VISIBILITY_MASK to force selection of
            // OpenMP, while the reverse would not be possible without this 
            // preference)
            if(dev_id.get_backend() == backend_id::omp)
              platform_score += 1;
            else
              platform_score += 2;
          } else if(dev->is_gpu()){
            // Always prefer GPU.
            // Note that we *add* scores, so a platform with more devices is
            // always preferred
            
            // Prefer CUDA, since a) CUDA tends to be the most reliable backend
            // and b) we know that the hardware is going to be a dGPU
            if(dev_id.get_backend() == backend_id::cuda)
              platform_score += 6;
            // HIP is typically a dGPU, but might also be an APU
            else if(dev_id.get_backend() == backend_id::hip)
              platform_score += 5;
            else {
              // OpenCL or L0 is most likely iGPU. Not many Intel dGPUs around.
              platform_score += 4;
            }
          } else {
            // not a CPU nor GPU? Such a device is currently not tested with
            // AdaptiveCpp, be cautious and prefer any other platform.
            platform_score += 0;
          }
        }

        if(platform_score > best_score) {
          best_backend = i;
          best_platform = j;
        }
      }
    }
  }

  if(best_score < 0) {
    HIPSYCL_DEBUG_ERROR << "[PCUDA] pcuda_thread_state: Did not find any "
                           "devices (not even CPU); this should "
                           "never happen. Things are going to break now."
                        << std::endl;
  } else {
    _current_backend = best_backend;
    _current_platform = best_platform;
    _current_device = 0;
  }


  _per_device_data.resize(topo.all_backends().size());
  for(int i = 0; i < _per_device_data.size(); ++i) {
    _per_device_data[i].resize(topo.get_backend(i)->platforms.size());
    for(int j = 0; j < _per_device_data[i].size(); ++j) {
      _per_device_data[i][j].resize(topo.get_platform(i, j)->devices.size());
    }
  }

}

int thread_local_state::get_device() const { return _current_device; }
int thread_local_state::get_platform() const { return _current_platform; }
int thread_local_state::get_backend() const { return _current_backend; }

internal_stream_t* thread_local_state::get_default_stream() const {
  assert(_current_backend < _per_device_data.size());
  assert(_current_platform < _per_device_data[_current_backend].size());
  assert(_current_device <
         _per_device_data[_current_backend][_current_platform].size());

  auto &device_data =
      _per_device_data[_current_backend][_current_platform][_current_device];


  if(internal_stream_t* s = device_data.default_stream.value_or(nullptr))
    return s;
  
  internal_stream_t* default_stream = nullptr;
  auto err = create_stream(default_stream, _rt, 0, 0);
  if(err != pcudaSuccess) {
    rt::register_error(rt::make_error(
        __acpp_here(),
        rt::error_info{"[PCUDA] default stream construction failed", err}));
    return nullptr;
  }
  assert(default_stream);
  device_data.default_stream = default_stream;

  return default_stream;
}


}
