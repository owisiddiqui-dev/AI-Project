# Environment Setup & Current Status

**Status: COMPLETE.** The CVA6 + Ara Verilator model builds and runs `hello_world` (SUCCESS, 0x54e cycles).

All sources and build outputs live on Drive D: in `D:\AI Thesis\ara`.

---

## System

- Host: Windows 11, WSL2 Ubuntu 22.04 (`wsl -d Ubuntu-22.04`), user `owi` (passwordless sudo).
- WSL VM has ~5.7 GB RAM; an 8 GB swapfile `/swapfile_extra` is used to avoid OOM during LLVM/Verilator builds.
- `Acquire::ForceIPv4 "true";` in `/etc/apt/apt.conf.d/99force-ipv4` fixes apt over WSL IPv6.
- `/mnt/d_ara` is a bind mount of `/mnt/d/AI Thesis/ara`. It exists because GNU Make breaks on the space in "AI Thesis".
  **It does not survive WSL restarts** (the fstab entry doesn't work), and the swapfile is also dropped.

## How to run anything (always use the wrapper)

`scripts/run_as_owi.sh` re-enables swap, re-creates the `/mnt/d_ara` mount, then runs a script as user `owi`
(running `make` as root breaks git with "dubious ownership"):

```powershell
wsl -d Ubuntu-22.04 -u root --cd /tmp -- bash "/mnt/d/AI Thesis/scripts/run_as_owi.sh" "/mnt/d/AI Thesis/scripts/run_sim.sh" hello_world
```

## Installed toolchain (`D:\AI Thesis\ara\install\`)

| Component | Result |
|---|---|
| Verilator | 5.047 (pinned commit 06263ec) |
| Spike | `riscv-isa-sim`, `riscv-isa-sim-mod` |
| LLVM/Clang | 20.1.0, RISC-V target, newlib, compiler-rt builtins |
| Bender | 0.31.0 (`hardware/bender`) |

## Problems hit and fixes (for reproducibility)

| Problem | Cause | Fix |
|---|---|---|
| Make circular dependency / bad configure | space in path | `/mnt/d_ara` bind mount |
| git "dubious ownership" | make run as root | run as `owi` via `run_as_owi.sh` |
| LLVM `Killed signal terminated program cc1plus` | OOM with default parallelism | resume with `ninja -j2` (`resume_llvm.sh`), swap on |
| compiler-rt "operating system not supported" | extra components pull in sanitizer_common | `build_compiler_rt.sh` (builtins only) |
| `bender update` axi version conflict | re-resolves versions | use `bender checkout` with committed `Bender.lock` |
| Link error `simutil_memload` / `simutil_set_mem`, then "No memory found at i_dram" | repo SRAM patch never applied | `make apply-patches` (`patch_and_verilate.sh`) |

## Rebuild from scratch (order)

1. `make verilator`, `make riscv-isa-sim` (as `owi`, in `/mnt/d_ara`)
2. `scripts/build_llvm.sh` / `scripts/resume_llvm.sh`, then `scripts/build_compiler_rt.sh`
3. `scripts/build_ara_rtl.sh` (Bender checkout, compile app, verilate), then `scripts/patch_and_verilate.sh`
4. `scripts/run_sim.sh <app>` to run an app, e.g. `hello_world`, `fconv2d`, `imatmul`

Default config: 4 lanes, VLEN=4096 (`config/default.mk`).
