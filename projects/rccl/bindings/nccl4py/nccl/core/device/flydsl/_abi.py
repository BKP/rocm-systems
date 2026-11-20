# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Entry points of ``librccl_device_dsl.bc`` (``bindings/ir/nccl_device_dsl.h``).

:data:`DEVICE_FUNCTIONS` maps each symbol to its ``(argument types, return
type)`` in ``flydsl.expr.extern.ffi`` names. The enums mirror the axes the
symbols are specialized on.
"""

from enum import IntEnum


class Coop(IntEnum):
    """Cooperative group that calls an operation together."""

    THREAD = 0
    WARP = 1
    CTA = 2


class Signal(IntEnum):
    """Signal action a GIN ``put`` or ``signal`` applies on the peer."""

    NONE = 0
    WEAK_INC = 1
    WEAK_ADD = 2
    STRONG_INC = 3
    STRONG_ADD = 4


class Team(IntEnum):
    """Team a peer index is numbered in; mirrors ``ncclDslTeam``."""

    WORLD = 0
    LSA = 1
    RAIL = 2


COOP_TAG = {Coop.THREAD: "Thread", Coop.WARP: "Warp", Coop.CTA: "Cta"}
SIGNAL_TAG = {
    Signal.NONE: "None",
    Signal.WEAK_INC: "WeakSignalInc",
    Signal.WEAK_ADD: "WeakSignalAdd",
    Signal.STRONG_INC: "StrongSignalInc",
    Signal.STRONG_ADD: "StrongSignalAdd",
}

_COMM = "uint64"
_WIN = "uint64"
_EFFECT = "int32"


def _device_functions() -> dict[str, tuple[tuple[str, ...], str]]:
    functions = {
        "ncclDslGinBackendMask": ((), "uint32"),
        "ncclDslDevCommSize": ((), "uint32"),
        "ncclDslVersion": ((), "uint32"),
        "ncclDslDevCommRank": ((_COMM,), "int32"),
        "ncclDslDevCommNRanks": ((_COMM,), "int32"),
        "ncclDslDevCommLsaRank": ((_COMM,), "int32"),
        "ncclDslDevCommLsaSize": ((_COMM,), "int32"),
        "ncclDslGinReadSignal": ((_COMM, "int32", "uint32"), "uint64"),
        "ncclDslGinResetSignal": ((_COMM, "int32", "uint32"), _EFFECT),
        "ncclDslGetLocalPointer": ((_WIN, "uint64"), "uint64"),
        "ncclDslGetLsaPointer": ((_WIN, "uint64", "int32"), "uint64"),
        "ncclDslGetPeerPointer": ((_WIN, "uint64", "int32"), "uint64"),
    }
    put = (
        _COMM,
        "int32",
        "uint32",
        "int32",
        _WIN,
        "uint64",
        _WIN,
        "uint64",
        "uint64",
        "uint32",
        "uint64",
    )
    signal = (_COMM, "int32", "uint32", "int32", "uint32", "uint64")
    for coop in COOP_TAG.values():
        for action in SIGNAL_TAG.values():
            functions[f"ncclDslGinPut_{coop}_{action}"] = (put, _EFFECT)
            if action != "None":
                functions[f"ncclDslGinSignal_{coop}_{action}"] = (signal, _EFFECT)
        functions[f"ncclDslGinFlush_{coop}"] = ((_COMM, "int32"), _EFFECT)
        functions[f"ncclDslGinWaitSignal_{coop}"] = ((_COMM, "int32", "uint32", "uint64"), _EFFECT)
        functions[f"ncclDslGinFlushAsync_{coop}"] = (
            (_COMM, "int32", "uint32", "int32", "uint64"),
            _EFFECT,
        )
        functions[f"ncclDslGinWait_{coop}"] = ((_COMM, "int32", "uint64"), _EFFECT)
        for barrier in ("LsaBarrierSync", "LsaBarrierArrive", "LsaBarrierWait"):
            functions[f"ncclDsl{barrier}_{coop}"] = ((_COMM, "uint32"), _EFFECT)
        for barrier in ("GinBarrierSync", "BarrierSync"):
            functions[f"ncclDsl{barrier}_{coop}"] = ((_COMM, "int32", "uint32", "uint32"), _EFFECT)
    return functions


DEVICE_FUNCTIONS = _device_functions()
