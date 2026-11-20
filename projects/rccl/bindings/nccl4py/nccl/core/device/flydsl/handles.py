# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Device-side handles for ``@flyc.kernel`` code.

``DevComm`` and ``Gin`` are FlyDSL structs carrying one ``Int64`` each, so a
handle built once at kernel entry flows through ``scf.if`` and ``scf.for``.
``coop``, ``signal`` and ``team`` select the entry point at trace time and must
be Python ints (the enum members); a ``Gin``'s context is a compile-time
constant too.
"""

import flydsl.expr as fx

from ._abi import COOP_TAG, SIGNAL_TAG, Coop, Signal, Team
from ._bindings import FUNCTIONS


def _const(value, name: str, allowed) -> int:
    """Return ``value`` if it is one of ``allowed``; it picks an entry point at trace time."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a Python int at trace time, not {type(value).__name__}")
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not one of {[int(v) for v in allowed]}")
    return value


def _coop(coop) -> str:
    return COOP_TAG[Coop(_const(coop, "coop", list(Coop)))]


def _team(team, allowed=tuple(Team)) -> int:
    return _const(team, "team", allowed)


@fx.struct
class Gin:
    """GIN context ``ctx`` of a device communicator, ``ncclGin(comm, ctx)``."""

    dev_comm: fx.Int64
    ctx: fx.Constexpr

    def put(
        self,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        *,
        signal=Signal.NONE,
        signal_id=0,
        signal_value=0,
        coop=Coop.THREAD,
        team=Team.WORLD,
    ):
        """Put ``nbytes`` of window ``src_win`` into ``peer``'s ``dst_win``, then ``signal``."""
        tag = SIGNAL_TAG[Signal(_const(signal, "signal", list(Signal)))]
        FUNCTIONS[f"ncclDslGinPut_{_coop(coop)}_{tag}"](
            self.dev_comm,
            self.ctx,
            _team(team),
            peer,
            dst_win,
            dst_off,
            src_win,
            src_off,
            nbytes,
            signal_id,
            signal_value,
        )

    def signal(
        self,
        peer,
        *,
        signal=Signal.WEAK_INC,
        signal_id=0,
        signal_value=0,
        coop=Coop.THREAD,
        team=Team.WORLD,
    ):
        """Apply ``signal`` to ``peer``'s signal ``signal_id`` with no data."""
        tag = SIGNAL_TAG[Signal(_const(signal, "signal", list(Signal)[1:]))]
        FUNCTIONS[f"ncclDslGinSignal_{_coop(coop)}_{tag}"](
            self.dev_comm, self.ctx, _team(team), peer, signal_id, signal_value
        )

    def flush(self, *, coop=Coop.THREAD):
        """Wait until the source buffers of this context's puts from ``coop`` may be reused."""
        FUNCTIONS[f"ncclDslGinFlush_{_coop(coop)}"](self.dev_comm, self.ctx)

    def read_signal(self, signal_id):
        """Current value of local signal ``signal_id``."""
        return FUNCTIONS["ncclDslGinReadSignal"](self.dev_comm, self.ctx, signal_id)

    def wait_signal(self, signal_id, least, *, coop=Coop.THREAD):
        """Wait until local signal ``signal_id`` reaches ``least``."""
        FUNCTIONS[f"ncclDslGinWaitSignal_{_coop(coop)}"](self.dev_comm, self.ctx, signal_id, least)

    def reset_signal(self, signal_id):
        """Set local signal ``signal_id`` to 0; it must not race with incoming signals."""
        FUNCTIONS["ncclDslGinResetSignal"](self.dev_comm, self.ctx, signal_id)

    def flush_async(self, peer, request, *, coop=Coop.THREAD, team=Team.WORLD):
        """Start a flush of this context's puts to ``peer``.

        ``request`` is the address of 16 bytes of memory for the request; pass
        it to :meth:`wait` with the same ``coop``.
        """
        FUNCTIONS[f"ncclDslGinFlushAsync_{_coop(coop)}"](
            self.dev_comm, self.ctx, _team(team), peer, request
        )

    def wait(self, request, *, coop=Coop.THREAD):
        """Wait for the flush :meth:`flush_async` started on ``request``."""
        FUNCTIONS[f"ncclDslGinWait_{_coop(coop)}"](self.dev_comm, self.ctx, request)


@fx.struct
class DevComm:
    """Handle for a device-resident ``ncclDevComm`` (see :func:`device_comm`)."""

    ptr: fx.Int64

    @property
    def rank(self):
        return FUNCTIONS["ncclDslDevCommRank"](self.ptr)

    @property
    def n_ranks(self):
        return FUNCTIONS["ncclDslDevCommNRanks"](self.ptr)

    @property
    def lsa_rank(self):
        return FUNCTIONS["ncclDslDevCommLsaRank"](self.ptr)

    @property
    def lsa_size(self):
        return FUNCTIONS["ncclDslDevCommLsaSize"](self.ptr)

    def gin(self, ctx=0) -> Gin:
        """GIN context ``ctx``, a compile-time constant."""
        return Gin(dev_comm=self.ptr, ctx=_const(ctx, "ctx", range(1 << 31)))


__all__ = ["DevComm", "Gin"]
