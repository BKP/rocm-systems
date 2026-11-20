# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

import os
import re
import subprocess

import pytest

pytest.importorskip("flydsl", reason="FlyDSL is required for nccl.core.device.flydsl")

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402

import nccl.core as nccl  # noqa: E402
import nccl.core.device.flydsl as rd  # noqa: E402
from cuda.core import Device, system  # noqa: E402
from nccl.core.device.flydsl import _abi, _bindings, runtime  # noqa: E402

_IR_TYPES = {"int32": "i32", "uint32": "i32", "uint64": "i64"}


@pytest.fixture(scope="module")
def bitcode():
    try:
        return _bindings.bitcode_path()
    except FileNotFoundError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def gpu():
    if system.get_num_devices() < 1:
        pytest.skip("requires a HIP device")
    Device(0).set_current()


def test_abi_table_matches_bitcode(bitcode):
    llvm_dis = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm", "bin", "llvm-dis")
    if not os.path.isfile(llvm_dis):
        pytest.skip(f"{llvm_dis} not found")
    ir = subprocess.run(
        [llvm_dis, bitcode, "-o", "-"], check=True, capture_output=True, text=True
    ).stdout
    defined = {
        name: (ret, tuple(a.split()[0] for a in args.split(",") if a.strip()))
        for ret, name, args in re.findall(
            r"^define (?:[a-z_]+ )*?(i32|i64) @(ncclDsl\w+)\(([^)]*)\)", ir, re.M
        )
    }
    expected = {
        name: (_IR_TYPES[ret], tuple(_IR_TYPES[a] for a in args))
        for name, (args, ret) in _abi.DEVICE_FUNCTIONS.items()
    }
    assert defined == expected


@flyc.kernel
def _all_entry_points_kernel(flag: fx.Int32):
    if flag == 1:
        for fn in _bindings.FUNCTIONS.values():
            fn(*([0] * len(fn.args)))


@flyc.jit
def _run_all_entry_points(flag: fx.Int32, stream=fx.Stream(None)):
    _all_entry_points_kernel(flag).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)


def test_every_entry_point_links(bitcode, gpu):
    _run_all_entry_points(0)


@flyc.kernel
def _runtime_coop_kernel(dc: fx.Int64, coop: fx.Int32):
    rd.DevComm(dc).gin(0).flush(coop=coop)


@flyc.jit
def _run_runtime_coop(dc: fx.Int64, coop: fx.Int32, stream=fx.Stream(None)):
    _runtime_coop_kernel(dc, coop).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)


@flyc.kernel
def _bad_signal_kernel(dc: fx.Int64):
    rd.DevComm(dc).gin(0).signal(1, signal=rd.Signal.NONE)


@flyc.jit
def _run_bad_signal(dc: fx.Int64, stream=fx.Stream(None)):
    _bad_signal_kernel(dc).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)


def test_symbol_axes_are_checked_at_trace_time(bitcode, gpu):
    with pytest.raises(TypeError, match="coop must be a Python int"):
        _run_runtime_coop(0, 1)
    with pytest.raises(ValueError, match="signal="):
        _run_bad_signal(0)


def test_device_comm_rejects_a_mismatched_bitcode(bitcode, gpu, monkeypatch):
    if os.environ.get("NCCL_CUMEM_ENABLE") != "1":
        pytest.skip("device communicators need NCCL_CUMEM_ENABLE=1")
    comm = nccl.Communicator.init(nranks=1, rank=0, unique_id=nccl.get_unique_id())
    try:
        resource = comm.create_dev_comm(nccl.NCCLDevCommRequirements())
        dc = rd.device_comm(resource)
        assert dc.ptr == resource.device_ptr != 0
        mask, size, version = runtime._bitcode_info(bitcode)
        monkeypatch.setattr(runtime, "_bitcode_info", lambda path: (mask, size, version + 1))
        with pytest.raises(RuntimeError, match="NCCL version"):
            rd.device_comm(resource)
    finally:
        comm.destroy()
