#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""LSA all-reduce: peer pointers plus an LSA barrier.

Every rank's kernel waits at an LSA barrier until all ranks have entered,
reads its share of each 16-byte pack from every rank's input region through
``Window.lsa_ptr``, sums in f32, and writes the result to its own output
region. A second barrier keeps a rank from finishing while its peers still
read its input.

USAGE (one node, any number of ranks):
mpirun -n 8 python 03_lsa_allreduce.py
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

NUM_ELEMS = 256 * 1024
NUM_PACKS = NUM_ELEMS // 4
THREADS = 256
IN_OFF = 0
OUT_OFF = NUM_ELEMS * 4
WIN_BYTES = 2 * NUM_ELEMS * 4


def _pack_ptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global, 16), fx.Uint64(addr)
    )


@flyc.kernel(known_block_size=[THREADS, 1, 1])
def allreduce_kernel(dev_comm: fx.Int64, win_handle: fx.Int64, nranks: fx.Constexpr):
    dc = rd.DevComm(dev_comm)
    win = rd.Window(win_handle)
    inputs = [_pack_ptr(win.lsa_ptr(peer, IN_OFF)) for peer in range(nranks)]
    out = _pack_ptr(win.local_ptr(OUT_OFF))
    dc.lsa_barrier_sync(0)
    for pack in range(fx.thread_idx.x, NUM_PACKS, THREADS):
        acc = fx.ptr_load(inputs[0] + pack * 4, result_type=fx.Vector.make_type(4, fx.Float32))
        for peer in fx.range_constexpr(1, nranks):
            acc = acc + fx.ptr_load(
                inputs[peer] + pack * 4, result_type=fx.Vector.make_type(4, fx.Float32)
            )
        fx.ptr_store(acc, out + pack * 4)
    dc.lsa_barrier_sync(0)


@flyc.jit
def run_allreduce(
    dev_comm: fx.Int64, win_handle: fx.Int64, nranks: fx.Constexpr, stream=fx.Stream(None)
):
    allreduce_kernel(dev_comm, win_handle, nranks).launch(
        grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream
    )


def main():
    comm_mpi = MPI.COMM_WORLD
    rank, nranks = comm_mpi.Get_rank(), comm_mpi.Get_size()
    if nranks < 2:
        print("ERROR: this example needs at least 2 ranks on one node")
        return 1

    Device(rank).set_current()
    unique_id = comm_mpi.bcast(nccl.get_unique_id() if rank == 0 else None, root=0)
    comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    buf = nccl.mem_alloc(WIN_BYTES)
    win = comm.register_window(buf)
    data = np.zeros(2 * NUM_ELEMS, dtype=np.float32)
    data[:NUM_ELEMS] = (rank + 1) * np.arange(1, NUM_ELEMS + 1, dtype=np.float32)
    hip.hipMemcpy(int(buf.handle), data, WIN_BYTES, hip.hipMemcpyKind.hipMemcpyHostToDevice)

    dc = rd.device_comm(comm.create_dev_comm(nccl.NCCLDevCommRequirements(lsa_barrier_count=1)))
    run_allreduce(dc.ptr, win.handle, nranks)
    hip.hipDeviceSynchronize()

    got = np.empty(NUM_ELEMS, dtype=np.float32)
    hip.hipMemcpy(
        got, int(buf.handle) + OUT_OFF, NUM_ELEMS * 4, hip.hipMemcpyKind.hipMemcpyDeviceToHost
    )
    expected = nranks * (nranks + 1) // 2 * np.arange(1, NUM_ELEMS + 1, dtype=np.float32)
    ok = bool(np.allclose(got, expected, rtol=1e-6))
    print(
        f"Rank {rank}: {'all-reduce verified' if ok else 'all-reduce MISMATCH'} ({NUM_ELEMS} f32)"
    )

    comm.destroy()
    ok = comm_mpi.allreduce(ok, op=MPI.LAND)
    if rank == 0:
        print("SUCCESS!" if ok else "FAILED!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
