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

NCCL_DEVICE_INLINE ncclWindow_t ncclDslWindow(uint64_t window) {
  return reinterpret_cast<ncclWindow_t>(window);
}

// The entry points are inlined into the caller, so a constant `flags` folds to one team.
NCCL_DEVICE_INLINE ncclTeam ncclDslTeamOf(ncclDevComm const& comm, uint32_t flags) {
  switch (flags) {
  case ncclDslTeamWorld:
    return ncclTeamWorld(comm);
  case ncclDslTeamLsa:
    return ncclTeamLsa(comm);
  case ncclDslTeamRail:
    return ncclTeamRail(comm);
  default:
    __builtin_trap();
  }
}

NCCL_DEVICE_INLINE ncclGin_None ncclDslActionNone(uint32_t, uint64_t) {
  return {};
}
NCCL_DEVICE_INLINE ncclGin_WeakSignalInc ncclDslActionWeakSignalInc(uint32_t signal, uint64_t) {
  return {signal};
}
NCCL_DEVICE_INLINE ncclGin_WeakSignalAdd ncclDslActionWeakSignalAdd(uint32_t signal, uint64_t value) {
  return {signal, value};
}
NCCL_DEVICE_INLINE ncclGin_StrongSignalInc ncclDslActionStrongSignalInc(uint32_t signal, uint64_t) {
  return {signal};
}
NCCL_DEVICE_INLINE ncclGin_StrongSignalAdd ncclDslActionStrongSignalAdd(uint32_t signal, uint64_t value) {
  return {signal, value};
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

#define NCCL_DSL_DEFINE_GIN_PUT(COOP, ACTION) \
  NCCL_DSL_API int ncclDslGinPut_##COOP##_##ACTION(uint64_t devComm, int context, uint32_t flags, int peer, \
                                                   uint64_t dstWindow, uint64_t dstOffset, uint64_t srcWindow, \
                                                   uint64_t srcOffset, uint64_t bytes, uint32_t signal, \
                                                   uint64_t value) { \
    ncclDevComm const& comm = ncclDslComm(devComm); \
    ncclGin(comm, context) \
      .put(ncclDslTeamOf(comm, flags), peer, ncclDslWindow(dstWindow), dstOffset, ncclDslWindow(srcWindow), srcOffset, \
           bytes, ncclDslAction##ACTION(signal, value), ncclGin_None{}, ncclCoop##COOP{}); \
    return 0; \
  }
#define NCCL_DSL_DEFINE_GIN_PUTS(_, COOP) \
  NCCL_DSL_DEFINE_GIN_PUT(COOP, None) \
  NCCL_DSL_SIGNAL_ACTIONS(NCCL_DSL_DEFINE_GIN_PUT, COOP)
NCCL_DSL_COOPS(NCCL_DSL_DEFINE_GIN_PUTS, _)

#define NCCL_DSL_DEFINE_GIN_SIGNAL(COOP, ACTION) \
  NCCL_DSL_API int ncclDslGinSignal_##COOP##_##ACTION(uint64_t devComm, int context, uint32_t flags, int peer, \
                                                      uint32_t signal, uint64_t value) { \
    ncclDevComm const& comm = ncclDslComm(devComm); \
    ncclGin(comm, context) \
      .signal(ncclDslTeamOf(comm, flags), peer, ncclDslAction##ACTION(signal, value), ncclCoop##COOP{}); \
    return 0; \
  }
#define NCCL_DSL_DEFINE_GIN_SIGNALS(_, COOP) NCCL_DSL_SIGNAL_ACTIONS(NCCL_DSL_DEFINE_GIN_SIGNAL, COOP)
NCCL_DSL_COOPS(NCCL_DSL_DEFINE_GIN_SIGNALS, _)

#define NCCL_DSL_DEFINE_GIN_COOP_OPS(_, COOP) \
  NCCL_DSL_API int ncclDslGinFlush_##COOP(uint64_t devComm, int context) { \
    ncclGin(ncclDslComm(devComm), context).flush(ncclCoop##COOP{}); \
    return 0; \
  } \
  NCCL_DSL_API int ncclDslGinWaitSignal_##COOP(uint64_t devComm, int context, uint32_t signal, uint64_t least) { \
    ncclGin(ncclDslComm(devComm), context).waitSignal(ncclCoop##COOP{}, signal, least); \
    return 0; \
  } \
  NCCL_DSL_API int ncclDslGinFlushAsync_##COOP(uint64_t devComm, int context, uint32_t flags, int peer, \
                                               uint64_t request) { \
    ncclDevComm const& comm = ncclDslComm(devComm); \
    ncclGin(comm, context) \
      .flushAsync(ncclDslTeamOf(comm, flags), peer, reinterpret_cast<ncclGinRequest_t*>(request), ncclCoop##COOP{}); \
    return 0; \
  } \
  NCCL_DSL_API int ncclDslGinWait_##COOP(uint64_t devComm, int context, uint64_t request) { \
    ncclGin(ncclDslComm(devComm), context).wait(*reinterpret_cast<ncclGinRequest_t*>(request), ncclCoop##COOP{}); \
    return 0; \
  }
NCCL_DSL_COOPS(NCCL_DSL_DEFINE_GIN_COOP_OPS, _)

NCCL_DSL_API uint64_t ncclDslGinReadSignal(uint64_t devComm, int context, uint32_t signal) {
  return ncclGin(ncclDslComm(devComm), context).readSignal(signal);
}
NCCL_DSL_API int ncclDslGinResetSignal(uint64_t devComm, int context, uint32_t signal) {
  ncclGin(ncclDslComm(devComm), context).resetSignal(signal);
  return 0;
}

NCCL_DSL_API uint64_t ncclDslGetLocalPointer(uint64_t window, uint64_t offset) {
  return reinterpret_cast<uint64_t>(ncclGetLocalPointer(ncclDslWindow(window), offset));
}
NCCL_DSL_API uint64_t ncclDslGetLsaPointer(uint64_t window, uint64_t offset, int lsaPeer) {
  return reinterpret_cast<uint64_t>(ncclGetLsaPointer(ncclDslWindow(window), offset, lsaPeer));
}
NCCL_DSL_API uint64_t ncclDslGetPeerPointer(uint64_t window, uint64_t offset, int peer) {
  return reinterpret_cast<uint64_t>(ncclGetPeerPointer(ncclDslWindow(window), offset, peer));
}

// LSA sessions load their epoch on construction and store it on destruction, so a session per call keeps
// consecutive calls with the same index in step.
#define NCCL_DSL_DEFINE_BARRIERS(_, COOP) \
  NCCL_DSL_API int ncclDslLsaBarrierSync_##COOP(uint64_t devComm, uint32_t index) { \
    ncclLsaBarrierSession<ncclCoop##COOP> bar(ncclCoop##COOP{}, ncclDslComm(devComm), ncclTeamTagLsa{}, index); \
    bar.sync(ncclCoop##COOP{}, cuda::memory_order_acq_rel); \
    return 0; \
  } \
  NCCL_DSL_API int ncclDslLsaBarrierArrive_##COOP(uint64_t devComm, uint32_t index) { \
    ncclLsaBarrierSession<ncclCoop##COOP> bar(ncclCoop##COOP{}, ncclDslComm(devComm), ncclTeamTagLsa{}, index); \
    bar.arrive(ncclCoop##COOP{}, cuda::memory_order_release); \
    return 0; \
  } \
  NCCL_DSL_API int ncclDslLsaBarrierWait_##COOP(uint64_t devComm, uint32_t index) { \
    ncclLsaBarrierSession<ncclCoop##COOP> bar(ncclCoop##COOP{}, ncclDslComm(devComm), ncclTeamTagLsa{}, index); \
    bar.wait(ncclCoop##COOP{}, cuda::memory_order_acquire); \
    return 0; \
  } \
  NCCL_DSL_API int ncclDslGinBarrierSync_##COOP(uint64_t devComm, int context, uint32_t flags, uint32_t index) { \
    ncclGin gin(ncclDslComm(devComm), context); \
    if (flags == ncclDslTeamWorld) ncclGinBarrier(ncclCoop##COOP{}, gin, ncclTeamTagWorld{}, index); \
    else if (flags == ncclDslTeamRail) ncclGinBarrier(ncclCoop##COOP{}, gin, ncclTeamTagRail{}, index); \
    else __builtin_trap(); \
    return 0; \
  } \
  NCCL_DSL_API int ncclDslBarrierSync_##COOP(uint64_t devComm, int context, uint32_t flags, uint32_t index) { \
    ncclGin gin(ncclDslComm(devComm), context); \
    if (flags == ncclDslTeamWorld) ncclBarrier(ncclCoop##COOP{}, ncclTeamTagWorld{}, gin, index); \
    else if (flags == ncclDslTeamRail) ncclBarrier(ncclCoop##COOP{}, ncclTeamTagRail{}, gin, index); \
    else __builtin_trap(); \
    return 0; \
  }
NCCL_DSL_COOPS(NCCL_DSL_DEFINE_BARRIERS, _)

#endif // _NCCL_DEVICE_DSL__IMPL_H_
