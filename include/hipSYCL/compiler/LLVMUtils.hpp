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
#ifndef HIPSYCL_SSCP_LLVMUTILS_H
#define HIPSYCL_SSCP_LLVMUTILS_H

#include "llvm/IR/DerivedTypes.h"

inline llvm::Type *getPointerType(llvm::Type *PointeeT, int AddressSpace) {
#if LLVM_VERSION_MAJOR < 16
  return llvm::PointerType::get(PointeeT, AddressSpace);
#else
  return llvm::PointerType::get(PointeeT->getContext(), AddressSpace);
#endif
}

#endif
