# FlyDSL device API examples

Runnable examples for `nccl.core.device.flydsl`, the FlyDSL binding over the
RCCL device API. Kernels call `librccl_device_dsl.bc`, which RCCL builds with
`-DNCCL_EMIT_LLVM_IR=ON`.

Install FlyDSL 0.5 or newer and mpi4py, and run one GPU per rank. GIN and
symmetric windows need `NCCL_CUMEM_ENABLE=1`:

```sh
pip install "flydsl>=0.5" mpi4py
```

The bitcode is resolved from `$RCCL_DEVICE_DSL_BC` first, then next to the
loaded `librccl.so`: `<libdir>/rccl/` for an installed RCCL and `lib/` for a
build tree.

## The examples

| File | Topic | Ranks | Needs |
|---|---|---|---|
| `01_gin_put.py` | 1 MiB `Gin.put` with a weak signal, waited on by the peer's kernel | **exactly 2** | GIN |

Run under MPI, one rank per GPU:

```sh
NCCL_CUMEM_ENABLE=1 NCCL_GIN_TYPE=7 mpirun -n 2 python 01_gin_put.py   # Anvil SDMA, one node
```

`NCCL_GIN_TYPE` selects the GIN backend for `01_gin_put.py`: `7` for Anvil
SDMA on one node, `6` for rocSHMEM GDA, `2` for the IB proxy (add
`NCCL_P2P_DISABLE=1 RCCL_ENABLE_INTRANET=1` on one node).
