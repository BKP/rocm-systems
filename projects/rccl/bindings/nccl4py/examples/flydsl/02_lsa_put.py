#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""LSA put: the kernel stores straight into a peer's window.

``Window.lsa_ptr`` returns the address of an LSA peer's window as mapped in
this process, so the kernel moves the data itself with ordinary loads and
stores. Rank 0 copies 1 MiB from its window into rank 1's; the host then
checks the payload.

USAGE (one node):
mpirun -n 2 python 02_lsa_put.py
"""

import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. Install with: pip install mpi4py")
    sys.exit(1)

import numpy as np

import flydsl.compiler as flyc
import flydsl.expr as fx

import nccl.core as nccl
import nccl.core.device.flydsl as rd
from cuda.core import Device
from hip import hip

NUM_ELEMS = 1024 * 1024 // 8
NBYTES = NUM_ELEMS * 8
NUM_PACKS = NBYTES // 16
THREADS = 256
DST_LSA_RANK = 1


def _pack_ptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16), fx.Uint64(addr)
    )


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def lsa_put_kernel(win_handle: fx.Int64):
    win = rd.Window(win_handle)
    src = _pack_ptr(win.local_ptr())
    dst = _pack_ptr(win.lsa_ptr(DST_LSA_RANK))
    for pack in range(fx.thread_idx.x, NUM_PACKS, THREADS):
        value = fx.ptr_load(src + pack * 4, result_type=fx.Vector.make_type(4, fx.Int32))
        fx.ptr_store(value, dst + pack * 4)


@flyc.jit
def run_lsa_put(win_handle: fx.Int64, stream=fx.Stream(None)):
    lsa_put_kernel(win_handle).launch(grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream)


def main():
    comm_mpi = MPI.COMM_WORLD
    rank, nranks = comm_mpi.Get_rank(), comm_mpi.Get_size()
    if nranks != 2:
        if rank == 0:
            print("ERROR: this example needs exactly 2 ranks on one node")
        return 1

    Device(rank).set_current()
    unique_id = comm_mpi.bcast(nccl.get_unique_id() if rank == 0 else None, root=0)
    comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    buf = nccl.mem_alloc(NBYTES)
    win = comm.register_window(buf)
    payload = np.arange(1, NUM_ELEMS + 1, dtype=np.uint64)
    hip.hipMemcpy(
        int(buf.handle),
        payload if rank == 0 else np.zeros_like(payload),
        NBYTES,
        hip.hipMemcpyKind.hipMemcpyHostToDevice,
    )

    comm_mpi.Barrier()
    if rank == 0:
        run_lsa_put(win.handle)
        hip.hipDeviceSynchronize()
    comm_mpi.Barrier()

    ok = True
    if rank == DST_LSA_RANK:
        got = np.empty_like(payload)
        hip.hipMemcpy(got, int(buf.handle), NBYTES, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
        ok = bool(np.array_equal(got, payload))
        print(f"Rank {rank}: {'payload verified' if ok else 'payload MISMATCH'} ({NBYTES} bytes)")

    comm.destroy()
    ok = comm_mpi.allreduce(ok, op=MPI.LAND)
    if rank == 0:
        print("SUCCESS!" if ok else "FAILED!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
