# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Static checks of librccl_device_dsl.bc against bindings/ir/nccl_device_dsl.h."""

import os
import re
import subprocess

import pytest

_C_TO_IR = {"int": "i32", "uint32_t": "i32", "uint64_t": "i64"}


def _declared(paths):
    out = subprocess.run(
        [os.path.join(paths.LLVM_BIN, "clang++"), "-E", "-P", "-x", "c++", "-DNCCL_DSL_API=",
         os.path.join(paths.IR_DIR, "nccl_device_dsl.h")],
        check=True, capture_output=True, text=True,
    ).stdout
    decls = {}
    for ret, name, args in re.findall(r"(\w+)\s+(ncclDsl\w+)\(([^)]*)\);", out):
        types = [a.split()[0] for a in args.split(",") if a.strip()]
        decls[name] = (_C_TO_IR[ret], [_C_TO_IR[t] for t in types])
    return decls


def _defined(paths, bitcode):
    ir = subprocess.run(
        [os.path.join(paths.LLVM_BIN, "llvm-dis"), bitcode, "-o", "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    defs = {}
    for ret, name, args in re.findall(r"^define (?:[a-z_]+ )*?(i32|i64|void) @(\w+)\(([^)]*)\)", ir, re.M):
        defs[name] = (ret, [a.split()[0] for a in args.split(",") if a.strip()])
    return ir, defs


@pytest.mark.dsl
def test_dsl_bitcode_defines_exactly_the_declared_entry_points(paths, dsl_bitcode):
    _, defs = _defined(paths, dsl_bitcode)
    exported = {name: sig for name, sig in defs.items() if name.startswith("ncclDsl")}
    assert exported == _declared(paths)


@pytest.mark.dsl
def test_dsl_bitcode_links_into_dsl_jits(paths, dsl_bitcode):
    ir, _ = _defined(paths, dsl_bitcode)
    assert '!"amdhsa_code_object_version", i32 600}' in ir
    assert "@llvm.used" not in ir
    assert not re.search(r"llvm\.amdgcn\.global\.(load|store)\.b128", ir)
