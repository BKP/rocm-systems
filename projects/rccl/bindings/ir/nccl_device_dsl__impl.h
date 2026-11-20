/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_DSL__IMPL_H_
#define _NCCL_DEVICE_DSL__IMPL_H_

// Inlined into the calling kernel; -fgpu-rdc keeps the definitions, which have no caller here.
#define NCCL_DSL_API extern "C" __device__ __attribute__((always_inline, visibility("default")))
#include "nccl_device.h"
#include "nccl_device_dsl.h"

namespace {
NCCL_DEVICE_INLINE ncclDevComm const& ncclDslComm(uint64_t devComm) {
  return *reinterpret_cast<ncclDevComm const*>(devComm);
}
} // namespace

NCCL_DSL_API uint32_t ncclDslGinBackendMask() {
  return NCCL_GIN_BACKEND_MASK_ALL;
}
NCCL_DSL_API uint32_t ncclDslDevCommSize() {
  return sizeof(ncclDevComm);
}
NCCL_DSL_API uint32_t ncclDslVersion() {
  return NCCL_VERSION_CODE;
}

NCCL_DSL_API int ncclDslDevCommRank(uint64_t devComm) {
  return ncclDslComm(devComm).rank;
}
NCCL_DSL_API int ncclDslDevCommNRanks(uint64_t devComm) {
  return ncclDslComm(devComm).nRanks;
}
NCCL_DSL_API int ncclDslDevCommLsaRank(uint64_t devComm) {
  return ncclDslComm(devComm).lsaRank;
}
NCCL_DSL_API int ncclDslDevCommLsaSize(uint64_t devComm) {
  return ncclDslComm(devComm).lsaSize;
}

#endif // _NCCL_DEVICE_DSL__IMPL_H_
