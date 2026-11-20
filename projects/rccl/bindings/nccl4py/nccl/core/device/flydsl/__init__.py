# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""FlyDSL bindings for the RCCL device API.

``@flyc.kernel`` code calls GIN operations from
``librccl_device_dsl.bc``, which RCCL builds with ``-DNCCL_EMIT_LLVM_IR=ON``.
Requires FlyDSL 0.5 or later. Handles cross the kernel boundary as ``Int64``::

    import nccl.core.device.flydsl as rd

    @flyc.kernel
    def put_kernel(dc: fx.Int64, send: fx.Int64, recv: fx.Int64, nbytes: fx.Int64):
        gin = rd.DevComm(dc).gin(0)
        if fx.thread_idx.x == 0:
            gin.put(1, recv, 0, send, 0, nbytes, signal=rd.Signal.WEAK_INC, signal_id=0)
        gin.flush(coop=rd.Coop.CTA)

    dc = rd.device_comm(comm.create_dev_comm(requirements))
    run_put(dc.ptr, send_window.handle, recv_window.handle, nbytes)

See ``examples/flydsl`` for complete programs.
"""

try:
    import flydsl.expr  # noqa: F401
except ImportError as e:
    raise ImportError("nccl.core.device.flydsl requires FlyDSL: pip install flydsl") from e

from ._abi import Coop, Signal, Team
from ._bindings import bitcode_path
from .handles import DevComm, Gin
from .runtime import DeviceComm, device_comm

__all__ = [
    "Coop",
    "DevComm",
    "DeviceComm",
    "Gin",
    "Signal",
    "Team",
    "bitcode_path",
    "device_comm",
]
