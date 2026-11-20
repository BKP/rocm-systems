/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Multi-rank tests for the scalar-argument entry points in librccl_device_dsl.bc.
 *
 * Run (2 ranks, one GPU each):
 *   NCCL_CUMEM_ENABLE=1 NCCL_GIN_TYPE=<2|6|7> mpirun -np 2 ./IR_dsl_mpi_test.exe
 ************************************************************************/

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <nccl.h>
#include <nccl_device.h>
#include "nccl_device_dsl.h"

#define HIP_EXPECT(stmt) EXPECT_EQ((stmt), hipSuccess)

namespace {

constexpr int kThreads = 256;
constexpr size_t kChunk = 4096;
constexpr int kSlots = 32;
constexpr uint64_t kAddValue = 5;
constexpr int kIters = 16;

enum Coop {
  kThread,
  kWarp,
  kCta,
};

enum Action {
  kNone,
  kWeakSignalInc,
  kWeakSignalAdd,
  kStrongSignalInc,
  kStrongSignalAdd,
};

#define COOP_SWITCH(coop, CALL) \
  switch (coop) { \
  case kThread: \
    CALL(Thread); \
    break; \
  case kWarp: \
    CALL(Warp); \
    break; \
  default: \
    CALL(Cta); \
  }

#define SIGNAL_SWITCH(action, CALL, COOP) \
  switch (action) { \
  case kWeakSignalInc: \
    CALL(COOP, WeakSignalInc); \
    break; \
  case kWeakSignalAdd: \
    CALL(COOP, WeakSignalAdd); \
    break; \
  case kStrongSignalInc: \
    CALL(COOP, StrongSignalInc); \
    break; \
  default: \
    CALL(COOP, StrongSignalAdd); \
  }

__device__ bool inCoop(int coop) {
  return coop == kCta || (coop == kWarp ? threadIdx.x < warpSize : threadIdx.x == 0);
}

bool isAdd(int action) {
  return action == kWeakSignalAdd || action == kStrongSignalAdd;
}

bool allRanks(bool ok) {
  int v = ok ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return v == 1;
}

int localRank() {
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int rank = 0;
  MPI_Comm_rank(node, &rank);
  MPI_Comm_free(&node);
  return rank;
}

__global__ void buildInfoKernel(uint32_t* out) {
  out[0] = ncclDslGinBackendMask() == NCCL_GIN_BACKEND_MASK_ALL;
  out[1] = ncclDslDevCommSize() == sizeof(ncclDevComm);
  out[2] = ncclDslVersion() == NCCL_VERSION_CODE;
}

__global__ void accessorKernel(uint64_t dc, int* out) {
  out[0] = ncclDslDevCommRank(dc);
  out[1] = ncclDslDevCommNRanks(dc);
  out[2] = ncclDslDevCommLsaRank(dc);
  out[3] = ncclDslDevCommLsaSize(dc);
}

__global__ void putKernel(uint64_t dc, uint64_t dstWin, uint64_t srcWin, int coop, int action, bool put, int slot) {
  uint64_t offset = slot * kChunk;
  uint64_t value = action == kWeakSignalAdd || action == kStrongSignalAdd ? kAddValue : 0;
  if (inCoop(coop)) {
#define PUT(COOP, ACTION) \
  ncclDslGinPut_##COOP##_##ACTION(dc, 0, ncclDslTeamWorld, 1, dstWin, offset, srcWin, offset, kChunk, slot, value)
#define SIGNAL(COOP, ACTION) ncclDslGinSignal_##COOP##_##ACTION(dc, 0, ncclDslTeamWorld, 1, slot, value)
#define PUT_NONE(COOP) PUT(COOP, None)
#define PUT_COOP(COOP) SIGNAL_SWITCH(action, PUT, COOP)
#define SIGNAL_COOP(COOP) SIGNAL_SWITCH(action, SIGNAL, COOP)
#define FLUSH(COOP) ncclDslGinFlush_##COOP(dc, 0)
    if (put && action == kNone) {
      COOP_SWITCH(coop, PUT_NONE)
    } else if (put) {
      COOP_SWITCH(coop, PUT_COOP)
    } else {
      COOP_SWITCH(coop, SIGNAL_COOP)
    }
    COOP_SWITCH(coop, FLUSH)
  }
  if (action == kNone && threadIdx.x == 0) ncclDslGinSignal_Thread_StrongSignalInc(dc, 0, ncclDslTeamWorld, 1, slot, 0);
}

__global__ void waitKernel(uint64_t dc, int coop, int slot, uint64_t least, uint64_t* seen) {
  if (inCoop(coop)) {
#define WAIT_SIGNAL(COOP) ncclDslGinWaitSignal_##COOP(dc, 0, slot, least)
    COOP_SWITCH(coop, WAIT_SIGNAL)
  }
  if (threadIdx.x == 0) {
    seen[0] = ncclDslGinReadSignal(dc, 0, slot);
    ncclDslGinResetSignal(dc, 0, slot);
    seen[1] = ncclDslGinReadSignal(dc, 0, slot);
  }
}

__global__ void flushAsyncKernel(uint64_t dc, uint64_t dstWin, uint64_t srcWin, int coop, uint64_t request) {
  if (threadIdx.x == 0) ncclDslGinPut_Thread_None(dc, 0, ncclDslTeamWorld, 1, dstWin, 0, srcWin, 0, kChunk, 0, 0);
  if (inCoop(coop)) {
#define FLUSH_ASYNC(COOP) ncclDslGinFlushAsync_##COOP(dc, 0, ncclDslTeamWorld, 1, request)
#define WAIT(COOP) ncclDslGinWait_##COOP(dc, 0, request)
    COOP_SWITCH(coop, FLUSH_ASYNC)
    COOP_SWITCH(coop, WAIT)
  }
  if (threadIdx.x == 0) ncclDslGinSignal_Thread_StrongSignalInc(dc, 0, ncclDslTeamWorld, 1, coop, 0);
}

__global__ void pointerKernel(uint64_t window, int lsaPeer, int peer, uint64_t* out) {
  ncclWindow_t w = reinterpret_cast<ncclWindow_t>(window);
  out[0] = ncclDslGetLocalPointer(window, 64) == reinterpret_cast<uint64_t>(ncclGetLocalPointer(w, 64));
  out[1] = ncclDslGetLsaPointer(window, 64, lsaPeer) == reinterpret_cast<uint64_t>(ncclGetLsaPointer(w, 64, lsaPeer));
  out[2] = ncclDslGetPeerPointer(window, 64, peer) == reinterpret_cast<uint64_t>(ncclGetPeerPointer(w, 64, peer));
}

__global__ void lsaStoreKernel(uint64_t window, int lsaPeer, uint64_t value) {
  *reinterpret_cast<uint64_t*>(ncclDslGetLsaPointer(window, 0, lsaPeer)) = value;
}

__global__ void lsaBarrierKernel(uint64_t dc, int coop) {
  if (!inCoop(coop)) return;
#define LSA_SYNC(COOP) ncclDslLsaBarrierSync_##COOP(dc, 0)
#define LSA_ARRIVE(COOP) ncclDslLsaBarrierArrive_##COOP(dc, 0)
#define LSA_WAIT(COOP) ncclDslLsaBarrierWait_##COOP(dc, 0)
  for (int i = 0; i < kIters; i++) {
    if (i % 2) {
      COOP_SWITCH(coop, LSA_ARRIVE)
      COOP_SWITCH(coop, LSA_WAIT)
    } else {
      COOP_SWITCH(coop, LSA_SYNC)
    }
  }
}

__global__ void ginBarrierKernel(uint64_t dc, int coop, uint32_t team, bool composite) {
  if (!inCoop(coop)) return;
#define GIN_BARRIER(COOP) ncclDslGinBarrierSync_##COOP(dc, 0, team, 0)
#define BARRIER(COOP) ncclDslBarrierSync_##COOP(dc, 0, team, 0)
  for (int i = 0; i < kIters; i++) {
    if (composite) {
      COOP_SWITCH(coop, BARRIER)
    } else {
      COOP_SWITCH(coop, GIN_BARRIER)
    }
  }
}

} // namespace

class IRDslMPITest : public ::testing::Test {
protected:
  int rank_ = -1, nRanks_ = -1;
  ncclComm_t comm_ = nullptr;
  void* src_ = nullptr;
  void* dst_ = nullptr;
  ncclWindow_t srcWin_ = nullptr, dstWin_ = nullptr;
  ncclDevComm devComm_{};
  bool devCommValid_ = false;
  ncclDevComm* dc_ = nullptr;

  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks_);
    if (nRanks_ != 2) GTEST_SKIP() << "needs exactly 2 ranks";
    ASSERT_EQ(hipSetDevice(localRank()), hipSuccess);
    ncclUniqueId id{};
    if (rank_ == 0) ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ASSERT_EQ(ncclCommInitRank(&comm_, nRanks_, id, rank_), ncclSuccess);
  }

  void TearDown() override {
    if (dc_) (void)hipFree(dc_);
    if (devCommValid_) (void)ncclDevCommDestroy(comm_, &devComm_);
    if (srcWin_) (void)ncclCommWindowDeregister(comm_, srcWin_);
    if (dstWin_) (void)ncclCommWindowDeregister(comm_, dstWin_);
    if (src_) (void)ncclMemFree(src_);
    if (dst_) (void)ncclMemFree(dst_);
    if (comm_) (void)ncclCommDestroy(comm_);
  }

  // Agreed across ranks, so a false result is a safe GTEST_SKIP.
  bool createDevComm(bool gin) {
    bool ok =
      ncclMemAlloc(&src_, kSlots * kChunk) == ncclSuccess && ncclMemAlloc(&dst_, kSlots * kChunk) == ncclSuccess;
    ok = ok && ncclCommWindowRegister(comm_, src_, kSlots * kChunk, &srcWin_, NCCL_WIN_COLL_SYMMETRIC) == ncclSuccess;
    ok = ok && ncclCommWindowRegister(comm_, dst_, kSlots * kChunk, &dstWin_, NCCL_WIN_COLL_SYMMETRIC) == ncclSuccess;
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = 1;
    if (gin) {
      reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
      reqs.ginSignalCount = kSlots;
      reqs.ginStrongSignalsRequired = true;
      reqs.railGinBarrierCount = 1;
      reqs.worldGinBarrierCount = 1;
    }
    devCommValid_ = ok && ncclDevCommCreate(comm_, &reqs, &devComm_) == ncclSuccess;
    ok = devCommValid_ && hipMalloc(&dc_, sizeof(ncclDevComm)) == hipSuccess &&
         hipMemcpy(dc_, &devComm_, sizeof(ncclDevComm), hipMemcpyHostToDevice) == hipSuccess;
    return allRanks(ok);
  }

  uint64_t dc() const {
    return reinterpret_cast<uint64_t>(dc_);
  }
  uint64_t win(ncclWindow_t w) const {
    return reinterpret_cast<uint64_t>(w);
  }

  std::vector<uint8_t> pattern(uint8_t seed) const {
    std::vector<uint8_t> bytes(kSlots * kChunk);
    for (size_t i = 0; i < bytes.size(); i++) bytes[i] = static_cast<uint8_t>(seed + i * 7);
    return bytes;
  }
};

TEST_F(IRDslMPITest, BuildInfo) {
  uint32_t* out = nullptr;
  ASSERT_EQ(hipMallocManaged(&out, 3 * sizeof(uint32_t)), hipSuccess);
  buildInfoKernel<<<1, 1>>>(out);
  HIP_EXPECT(hipDeviceSynchronize());
  EXPECT_EQ(out[0], 1u) << "GIN backend mask";
  EXPECT_EQ(out[1], 1u) << "sizeof(ncclDevComm)";
  EXPECT_EQ(out[2], 1u) << "NCCL_VERSION_CODE";
  HIP_EXPECT(hipFree(out));
}

TEST_F(IRDslMPITest, DevCommAccessors) {
  if (!createDevComm(false)) GTEST_SKIP() << "ncclDevCommCreate failed";
  int* out = nullptr;
  ASSERT_EQ(hipMallocManaged(&out, 4 * sizeof(int)), hipSuccess);
  accessorKernel<<<1, 1>>>(dc(), out);
  HIP_EXPECT(hipDeviceSynchronize());
  EXPECT_EQ(out[0], devComm_.rank);
  EXPECT_EQ(out[1], devComm_.nRanks);
  EXPECT_EQ(out[2], devComm_.lsaRank);
  EXPECT_EQ(out[3], devComm_.lsaSize);
  HIP_EXPECT(hipFree(out));
}

TEST_F(IRDslMPITest, GinPutAndSignal) {
  if (!createDevComm(true)) GTEST_SKIP() << "GIN ncclDevCommCreate with strong signals failed";
  uint64_t* seen = nullptr;
  ASSERT_EQ(hipMallocManaged(&seen, 2 * sizeof(uint64_t)), hipSuccess);
  std::vector<uint8_t> src = pattern(0x11);
  HIP_EXPECT(hipMemcpy(src_, src.data(), src.size(), hipMemcpyHostToDevice));
  HIP_EXPECT(hipMemset(dst_, 0, kSlots * kChunk));
  MPI_Barrier(MPI_COMM_WORLD);
  int slot = 0;
  std::vector<int> putSlots;
  for (int put = 1; put >= 0; put--) {
    for (int coop = kThread; coop <= kCta; coop++) {
      for (int action = put ? kNone : kWeakSignalInc; action <= kStrongSignalAdd; action++, slot++) {
        uint64_t least = isAdd(action) ? kAddValue : 1;
        if (rank_ == 0) putKernel<<<1, kThreads>>>(dc(), win(dstWin_), win(srcWin_), coop, action, put, slot);
        else waitKernel<<<1, kThreads>>>(dc(), coop, slot, least, seen);
        HIP_EXPECT(hipDeviceSynchronize());
        if (rank_ == 1) {
          EXPECT_EQ(seen[0], least) << "put " << put << " coop " << coop << " action " << action;
          EXPECT_EQ(seen[1], 0u) << "put " << put << " coop " << coop << " action " << action;
        }
        if (put) putSlots.push_back(slot);
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
  if (rank_ == 1) {
    std::vector<uint8_t> got(kSlots * kChunk);
    HIP_EXPECT(hipMemcpy(got.data(), dst_, got.size(), hipMemcpyDeviceToHost));
    for (int s : putSlots) EXPECT_EQ(memcmp(&got[s * kChunk], &src[s * kChunk], kChunk), 0) << "slot " << s;
  }
  HIP_EXPECT(hipFree(seen));
}

TEST_F(IRDslMPITest, GinFlushAsyncWait) {
  if (!createDevComm(true)) GTEST_SKIP() << "GIN ncclDevCommCreate with strong signals failed";
  ncclGinRequest_t* request = nullptr;
  uint64_t* seen = nullptr;
  ASSERT_EQ(hipMalloc(&request, sizeof(ncclGinRequest_t)), hipSuccess);
  ASSERT_EQ(hipMallocManaged(&seen, 2 * sizeof(uint64_t)), hipSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int coop = kThread; coop <= kCta; coop++) {
    if (rank_ == 0)
      flushAsyncKernel<<<1, kThreads>>>(dc(), win(dstWin_), win(srcWin_), coop, reinterpret_cast<uint64_t>(request));
    else waitKernel<<<1, kThreads>>>(dc(), kThread, coop, 1, seen);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess) << "coop " << coop;
    MPI_Barrier(MPI_COMM_WORLD);
  }
  HIP_EXPECT(hipFree(request));
  HIP_EXPECT(hipFree(seen));
}

TEST_F(IRDslMPITest, LsaPointers) {
  if (!createDevComm(false)) GTEST_SKIP() << "ncclDevCommCreate failed";
  uint64_t* out = nullptr;
  ASSERT_EQ(hipMallocManaged(&out, 3 * sizeof(uint64_t)), hipSuccess);
  int lsaPeer = devComm_.lsaSize - 1 - devComm_.lsaRank;
  pointerKernel<<<1, 1>>>(win(dstWin_), lsaPeer, 1 - rank_, out);
  HIP_EXPECT(hipDeviceSynchronize());
  EXPECT_EQ(out[0], 1u) << "local pointer";
  EXPECT_EQ(out[1], 1u) << "LSA pointer";
  EXPECT_EQ(out[2], 1u) << "peer pointer";
  if (allRanks(devComm_.lsaSize == 2)) {
    HIP_EXPECT(hipMemset(dst_, 0, kChunk));
    MPI_Barrier(MPI_COMM_WORLD);
    lsaStoreKernel<<<1, 1>>>(win(dstWin_), lsaPeer, 0xC0FFEE00u + rank_);
    HIP_EXPECT(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t got = 0;
    HIP_EXPECT(hipMemcpy(&got, dst_, sizeof(got), hipMemcpyDeviceToHost));
    EXPECT_EQ(got, 0xC0FFEE00u + (1 - rank_));
  }
  HIP_EXPECT(hipFree(out));
}

TEST_F(IRDslMPITest, LsaBarrier) {
  if (!createDevComm(false)) GTEST_SKIP() << "ncclDevCommCreate failed";
  for (int coop = kThread; coop <= kCta; coop++) {
    lsaBarrierKernel<<<1, kThreads>>>(dc(), coop);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess) << "coop " << coop;
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TEST_F(IRDslMPITest, GinBarriers) {
  if (!createDevComm(true)) GTEST_SKIP() << "GIN ncclDevCommCreate with strong signals failed";
  for (int composite = 0; composite <= 1; composite++) {
    for (uint32_t team : {ncclDslTeamWorld, ncclDslTeamRail}) {
      for (int coop = kThread; coop <= kCta; coop++) {
        ginBarrierKernel<<<1, kThreads>>>(dc(), coop, team, composite);
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess)
          << "composite " << composite << " team " << team << " coop " << coop;
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}

namespace {
class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) MPI_Init(nullptr, nullptr);
  }
  void TearDown() override {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
  }
};

const ::testing::Environment* const kMpiEnv = ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
} // namespace
