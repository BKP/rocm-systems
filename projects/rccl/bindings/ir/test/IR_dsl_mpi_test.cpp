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

#include <nccl.h>
#include <nccl_device.h>
#include "nccl_device_dsl.h"

#define HIP_EXPECT(stmt) EXPECT_EQ((stmt), hipSuccess)

namespace {

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

} // namespace

class IRDslMPITest : public ::testing::Test {
protected:
  int rank_ = -1, nRanks_ = -1;
  ncclComm_t comm_ = nullptr;
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
    if (comm_) (void)ncclCommDestroy(comm_);
  }

  // Agreed across ranks, so a false result is a safe GTEST_SKIP.
  bool createDevComm() {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    devCommValid_ = ncclDevCommCreate(comm_, &reqs, &devComm_) == ncclSuccess;
    bool ok = devCommValid_ && hipMalloc(&dc_, sizeof(ncclDevComm)) == hipSuccess &&
              hipMemcpy(dc_, &devComm_, sizeof(ncclDevComm), hipMemcpyHostToDevice) == hipSuccess;
    return allRanks(ok);
  }

  uint64_t dc() const {
    return reinterpret_cast<uint64_t>(dc_);
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
  if (!createDevComm()) GTEST_SKIP() << "ncclDevCommCreate failed";
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
