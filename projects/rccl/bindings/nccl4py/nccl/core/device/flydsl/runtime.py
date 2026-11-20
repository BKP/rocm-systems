# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Host side: the device-resident ``ncclDevComm`` a kernel receives."""

import ctypes
import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

from nccl._hip_compat.cuda_core_shim._hip import check_hip, hip
import nccl.bindings.nccl as _nccl_bindings
from nccl.core.resources import DevCommResource

from ._bindings import FUNCTIONS, bitcode_path


def _store_u32(base, index, value):
    ptr = llvm.inttoptr(
        ir.Type.parse("!llvm.ptr<1>"), (fx.Int64(base) + fx.Int64(4 * index)).ir_value()
    )
    llvm.store(value, ptr, alignment=4)


@flyc.kernel
def _bitcode_info_kernel(out: fx.Int64):
    if fx.thread_idx.x == 0:
        _store_u32(out, 0, FUNCTIONS["ncclDslGinBackendMask"]())
        _store_u32(out, 1, FUNCTIONS["ncclDslDevCommSize"]())
        _store_u32(out, 2, FUNCTIONS["ncclDslVersion"]())


@flyc.jit
def _run_bitcode_info(out: fx.Int64, stream=fx.Stream(None)):
    _bitcode_info_kernel(out).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)


@functools.cache
def _bitcode_info(path: str) -> tuple[int, int, int]:
    """``(GIN backend mask, sizeof(ncclDevComm), NCCL version)`` of the bitcode at ``path``."""
    out = check_hip(hip.hipMalloc(12), "hipMalloc")
    try:
        _run_bitcode_info(int(out))
        host = (ctypes.c_uint32 * 3)()
        check_hip(
            hip.hipMemcpy(host, out, 12, hip.hipMemcpyKind.hipMemcpyDeviceToHost), "hipMemcpy"
        )
    finally:
        check_hip(hip.hipFree(out), "hipFree")
    return tuple(host)


def _check_bitcode(dev_comm: _nccl_bindings.DevComm) -> None:
    path = bitcode_path()
    mask, size, version = _bitcode_info(path)
    expected_size = _nccl_bindings.dev_comm_dtype.itemsize
    problems = []
    if size != expected_size:
        problems.append(f"sizeof(ncclDevComm) is {size}, nccl4py expects {expected_size}")
    if version != dev_comm.version:
        problems.append(
            f"built for NCCL version {version}, the ncclDevComm has version {dev_comm.version}"
        )
    for backend in dev_comm.gin_net_device_types[: dev_comm.gin_connection_count]:
        if not mask >> int(backend) & 1:
            problems.append(
                f"the ncclDevComm uses GIN backend {int(backend)}, not in its mask {mask:#x}"
            )
    if problems:
        raise RuntimeError(f"{path} does not match the loaded librccl.so: " + "; ".join(problems))


@dataclass(frozen=True)
class DeviceComm:
    """A device-resident ``ncclDevComm`` copy, owned by its :class:`~nccl.core.DevCommResource`."""

    ptr: int
    """Device address to pass to a kernel's ``Int64`` argument for :class:`DevComm`."""


def device_comm(resource: DevCommResource) -> DeviceComm:
    """Return the device-resident copy of ``resource``'s ``ncclDevComm``.

    Checks once per bitcode that ``librccl_device_dsl.bc`` was built for the
    same ``ncclDevComm`` layout, version and GIN backends as the loaded
    ``librccl.so``.

    Raises:
        RuntimeError: If the bitcode does not match.
    """
    _check_bitcode(resource.dev_comm)
    return DeviceComm(ptr=resource.device_ptr)


__all__ = ["DeviceComm", "device_comm"]
