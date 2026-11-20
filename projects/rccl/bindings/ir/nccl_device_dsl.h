/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_DSL_H_
#define _NCCL_DEVICE_DSL_H_

/*
 * Device API entry points for DSLs whose foreign-function interface passes only
 * integer scalars, built into librccl_device_dsl.bc.
 *
 * ncclDevComm* and ncclWindow_t cross as uint64_t. The ncclDevComm must be a
 * device-resident copy that outlives every kernel using it. Template arguments
 * of the device API are part of the symbol name, one symbol per value: the
 * cooperative group (Thread, Warp, Cta) and the GIN signal action. The team a
 * peer is numbered in is passed in a `flags` word (ncclDslTeam); any other
 * value traps. Functions called only for their effect return 0.
 */

#include <stdint.h>

#ifndef NCCL_DSL_API
#define NCCL_DSL_API extern "C" __device__
#endif

typedef enum {
  ncclDslTeamWorld = 0,
  ncclDslTeamLsa = 1,
  ncclDslTeamRail = 2,
} ncclDslTeam;

#define NCCL_DSL_COOPS(X, ...) \
  X(__VA_ARGS__, Thread) \
  X(__VA_ARGS__, Warp) \
  X(__VA_ARGS__, Cta)

#define NCCL_DSL_SIGNAL_ACTIONS(X, ...) \
  X(__VA_ARGS__, WeakSignalInc) \
  X(__VA_ARGS__, WeakSignalAdd) \
  X(__VA_ARGS__, StrongSignalInc) \
  X(__VA_ARGS__, StrongSignalAdd)

NCCL_DSL_API uint32_t ncclDslGinBackendMask();
NCCL_DSL_API uint32_t ncclDslDevCommSize();
NCCL_DSL_API uint32_t ncclDslVersion();

NCCL_DSL_API int ncclDslDevCommRank(uint64_t devComm);
NCCL_DSL_API int ncclDslDevCommNRanks(uint64_t devComm);
NCCL_DSL_API int ncclDslDevCommLsaRank(uint64_t devComm);
NCCL_DSL_API int ncclDslDevCommLsaSize(uint64_t devComm);

#define NCCL_DSL_DECLARE_GIN_PUT(COOP, ACTION) \
  NCCL_DSL_API int ncclDslGinPut_##COOP##_##ACTION(uint64_t devComm, int context, uint32_t flags, int peer, \
                                                   uint64_t dstWindow, uint64_t dstOffset, uint64_t srcWindow, \
                                                   uint64_t srcOffset, uint64_t bytes, uint32_t signal, \
                                                   uint64_t value);
#define NCCL_DSL_DECLARE_GIN_PUTS(_, COOP) \
  NCCL_DSL_DECLARE_GIN_PUT(COOP, None) \
  NCCL_DSL_SIGNAL_ACTIONS(NCCL_DSL_DECLARE_GIN_PUT, COOP)
NCCL_DSL_COOPS(NCCL_DSL_DECLARE_GIN_PUTS, _)

#define NCCL_DSL_DECLARE_GIN_SIGNAL(COOP, ACTION) \
  NCCL_DSL_API int ncclDslGinSignal_##COOP##_##ACTION(uint64_t devComm, int context, uint32_t flags, int peer, \
                                                      uint32_t signal, uint64_t value);
#define NCCL_DSL_DECLARE_GIN_SIGNALS(_, COOP) NCCL_DSL_SIGNAL_ACTIONS(NCCL_DSL_DECLARE_GIN_SIGNAL, COOP)
NCCL_DSL_COOPS(NCCL_DSL_DECLARE_GIN_SIGNALS, _)

#define NCCL_DSL_DECLARE_GIN_COOP_OPS(_, COOP) \
  NCCL_DSL_API int ncclDslGinFlush_##COOP(uint64_t devComm, int context); \
  NCCL_DSL_API int ncclDslGinWaitSignal_##COOP(uint64_t devComm, int context, uint32_t signal, uint64_t least); \
  NCCL_DSL_API int ncclDslGinFlushAsync_##COOP(uint64_t devComm, int context, uint32_t flags, int peer, \
                                               uint64_t request); \
  NCCL_DSL_API int ncclDslGinWait_##COOP(uint64_t devComm, int context, uint64_t request);
NCCL_DSL_COOPS(NCCL_DSL_DECLARE_GIN_COOP_OPS, _)

NCCL_DSL_API uint64_t ncclDslGinReadSignal(uint64_t devComm, int context, uint32_t signal);
NCCL_DSL_API int ncclDslGinResetSignal(uint64_t devComm, int context, uint32_t signal);

NCCL_DSL_API uint64_t ncclDslGetLocalPointer(uint64_t window, uint64_t offset);
NCCL_DSL_API uint64_t ncclDslGetLsaPointer(uint64_t window, uint64_t offset, int lsaPeer);
NCCL_DSL_API uint64_t ncclDslGetPeerPointer(uint64_t window, uint64_t offset, int peer);

#define NCCL_DSL_DECLARE_BARRIERS(_, COOP) \
  NCCL_DSL_API int ncclDslLsaBarrierSync_##COOP(uint64_t devComm, uint32_t index); \
  NCCL_DSL_API int ncclDslLsaBarrierArrive_##COOP(uint64_t devComm, uint32_t index); \
  NCCL_DSL_API int ncclDslLsaBarrierWait_##COOP(uint64_t devComm, uint32_t index); \
  NCCL_DSL_API int ncclDslGinBarrierSync_##COOP(uint64_t devComm, int context, uint32_t flags, uint32_t index); \
  NCCL_DSL_API int ncclDslBarrierSync_##COOP(uint64_t devComm, int context, uint32_t flags, uint32_t index);
NCCL_DSL_COOPS(NCCL_DSL_DECLARE_BARRIERS, _)

#endif // _NCCL_DEVICE_DSL_H_
