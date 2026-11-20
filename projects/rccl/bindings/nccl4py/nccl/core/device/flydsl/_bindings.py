# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""FlyDSL ``ffi`` callables for every entry point of ``librccl_device_dsl.bc``.

Each callable links the bitcode on first use, so importing needs no bitcode.
"""

import functools
import os

from flydsl.compiler.extern_link import link_extern
from flydsl.expr.extern import ffi

import nccl.bindings.nccl as _nccl_bindings
from nccl._show_versions import _resolve_so_path

from ._abi import DEVICE_FUNCTIONS

_BITCODE = "librccl_device_dsl.bc"


@functools.cache
def bitcode_path() -> str:
    """Locate ``librccl_device_dsl.bc`` for the loaded ``librccl.so``.

    Resolution order:

        1. ``$RCCL_DEVICE_DSL_BC``.
        2. ``<libdir>/rccl/librccl_device_dsl.bc``, the install location next to
           ``<libdir>/librccl.so``.
        3. ``<build>/lib/librccl_device_dsl.bc``, next to ``<build>/librccl.so``
           in an RCCL build tree.

    Raises:
        FileNotFoundError: If no candidate exists.
    """
    env = os.environ.get("RCCL_DEVICE_DSL_BC")
    if env:
        if not os.path.isfile(env):
            raise FileNotFoundError(f"RCCL_DEVICE_DSL_BC={env} does not exist")
        return env
    _nccl_bindings.get_version()
    lib = _resolve_so_path("librccl.so")
    candidates = (
        [] if lib is None else [lib.parent / "rccl" / _BITCODE, lib.parent / "lib" / _BITCODE]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"{_BITCODE} not found (searched {[str(c) for c in candidates]}). Build RCCL with "
        "-DNCCL_EMIT_LLVM_IR=ON or set RCCL_DEVICE_DSL_BC."
    )


class _Extern:
    """An ``ffi`` callable that links the bitcode on its first call."""

    def __init__(self, symbol: str, args: tuple[str, ...], ret: str):
        self.symbol, self.args, self.ret = symbol, args, ret
        self._fn = None

    def __call__(self, *args):
        if self._fn is None:
            self._fn = link_extern(
                ffi(self.symbol, list(self.args), self.ret), bitcode_path=bitcode_path()
            )
        return self._fn(*args)


FUNCTIONS = {symbol: _Extern(symbol, args, ret) for symbol, (args, ret) in DEVICE_FUNCTIONS.items()}

__all__ = ["FUNCTIONS", "bitcode_path"]
