# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""librccl_device_dsl.bc entry points on a live 2-rank communicator (bindings/ir/test/IR_dsl_mpi_test.cpp)."""

import pytest

_CASES = [
    "BuildInfo",
    "DevCommAccessors",
    "GinPutAndSignal",
    "GinFlushAsyncWait",
]


@pytest.mark.dsl
@pytest.mark.mpi
@pytest.mark.parametrize("case", _CASES)
def test_dsl_device(run_dsl_mpi_gtest, summarize_gtest, case):
    proc, log = run_dsl_mpi_gtest(f"IRDslMPITest.{case}", f"ir_dsl_{case}.log")
    summarize_gtest(proc, log, case)
