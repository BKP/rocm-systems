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
 * ncclDevComm* crosses as uint64_t; the ncclDevComm must be a device-resident
 * copy that outlives every kernel using it.
 */

#include <stdint.h>

#ifndef NCCL_DSL_API
#define NCCL_DSL_API extern "C" __device__
#endif

NCCL_DSL_API uint32_t ncclDslGinBackendMask();
NCCL_DSL_API uint32_t ncclDslDevCommSize();
NCCL_DSL_API uint32_t ncclDslVersion();

NCCL_DSL_API int ncclDslDevCommRank(uint64_t devComm);
NCCL_DSL_API int ncclDslDevCommNRanks(uint64_t devComm);
NCCL_DSL_API int ncclDslDevCommLsaRank(uint64_t devComm);
NCCL_DSL_API int ncclDslDevCommLsaSize(uint64_t devComm);

#endif // _NCCL_DEVICE_DSL_H_
