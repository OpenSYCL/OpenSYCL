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

#include "hipSYCL/compiler/sscp/pcuda/FreeKernelCall.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

void handleKernelCall(llvm::CallBase* CB) {

}

llvm::PreservedAnalyses FreeKernelCallPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  llvm::SmallPtrSet<llvm::Function *, 16> FreeKernels;

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
    if (Annotation.compare("acpp_free_kernel") == 0) {
      FreeKernels.insert(F);
    }
  });

  llvm::SmallPtrSet<llvm::CallBase*, 16> KernelCalls;
  for(auto* F: FreeKernels) {
    for(auto* U: F->users()) {
      if(auto* CB = llvm::dyn_cast<llvm::CallBase>(U)) {
        if(CB->getCalledFunction() == F) {
          KernelCalls.insert(CB);
        }
      }
    }
  }

  for(auto* C : KernelCalls)
    handleKernelCall(C);


  return llvm::PreservedAnalyses::none();
}
}
}

