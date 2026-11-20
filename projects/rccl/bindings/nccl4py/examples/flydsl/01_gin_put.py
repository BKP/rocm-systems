#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""GIN put with a completion signal, from FlyDSL kernels.

Rank 0's kernel puts 1 MiB from its send window into rank 1's receive window
and increments rank 1's signal 0. Rank 1's kernel waits for the signal; the
host then checks the payload.

USAGE:
mpirun -n 2 python 01_gin_put.py
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
DST_RANK = 1
SIGNAL_ID = 0


@flyc.kernel
def put_kernel(dev_comm: fx.Int64, send_win: fx.Int64, recv_win: fx.Int64, nbytes: fx.Int64):
    gin = rd.DevComm(dev_comm).gin(0)
    if fx.thread_idx.x == 0:
        gin.put(
            DST_RANK,
            recv_win,
            0,
            send_win,
            0,
            nbytes,
            signal=rd.Signal.WEAK_INC,
            signal_id=SIGNAL_ID,
        )
    gin.flush(coop=rd.Coop.CTA)


@flyc.kernel
def wait_kernel(dev_comm: fx.Int64):
    gin = rd.DevComm(dev_comm).gin(0)
    if fx.thread_idx.x == 0:
        gin.wait_signal(SIGNAL_ID, 1)


@flyc.jit
def run_put(
    dev_comm: fx.Int64,
    send_win: fx.Int64,
    recv_win: fx.Int64,
    nbytes: fx.Int64,
    stream=fx.Stream(None),
):
    put_kernel(dev_comm, send_win, recv_win, nbytes).launch(
        grid=(1, 1, 1), block=[64, 1, 1], stream=stream
    )


@flyc.jit
def run_wait(dev_comm: fx.Int64, stream=fx.Stream(None)):
    wait_kernel(dev_comm).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)


def main():
    comm_mpi = MPI.COMM_WORLD
    rank, nranks = comm_mpi.Get_rank(), comm_mpi.Get_size()
    if nranks != 2:
        if rank == 0:
            print("ERROR: this example needs exactly 2 ranks")
        return 1

    Device(rank).set_current()
    unique_id = comm_mpi.bcast(nccl.get_unique_id() if rank == 0 else None, root=0)
    comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    send = nccl.mem_alloc(NBYTES)
    recv = nccl.mem_alloc(NBYTES)
    send_win = comm.register_window(send)
    recv_win = comm.register_window(recv)
    payload = np.arange(1, NUM_ELEMS + 1, dtype=np.uint64)
    hip.hipMemcpy(int(send.handle), payload, NBYTES, hip.hipMemcpyKind.hipMemcpyHostToDevice)
    hip.hipMemcpy(
        int(recv.handle), np.zeros_like(payload), NBYTES, hip.hipMemcpyKind.hipMemcpyHostToDevice
    )

    reqs = nccl.NCCLDevCommRequirements(
        gin_connection_type=nccl.NcclGinConnectionType.FULL, gin_signal_count=1
    )
    dc = rd.device_comm(comm.create_dev_comm(reqs))
    comm_mpi.Barrier()
    if rank == 0:
        run_put(dc.ptr, send_win.handle, recv_win.handle, NBYTES)
    else:
        run_wait(dc.ptr)
    hip.hipDeviceSynchronize()
    comm_mpi.Barrier()

    ok = True
    if rank == DST_RANK:
        got = np.empty_like(payload)
        hip.hipMemcpy(got, int(recv.handle), NBYTES, hip.hipMemcpyKind.hipMemcpyDeviceToHost)
        ok = bool(np.array_equal(got, payload))
        print(f"Rank {rank}: {'payload verified' if ok else 'payload MISMATCH'} ({NBYTES} bytes)")

    comm.destroy()
    ok = comm_mpi.allreduce(ok, op=MPI.LAND)
    if rank == 0:
        print("SUCCESS!" if ok else "FAILED!")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
