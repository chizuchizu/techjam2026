#!/usr/bin/env python3
"""tp_elf.py - static memory accounting straight from the linked firmware.

Every SRAM and flash number in this repo's optimisation logs was hand-copied
from PlatformIO's build summary into markdown. That is exactly how the
"768 KB/forward" style figure drifts from reality without anyone noticing. This
module reads the ELF instead, so the memory half of a tinyprof artifact is
derived, reproducible, and diffable.

Uses only the toolchain binutils that PlatformIO already installed - no new
Python dependency, in line with CONTRIBUTING.md's stdlib-Python preference.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess

DEFAULT_TOOLCHAIN = pathlib.Path.home() / ".platformio/packages/toolchain-riscv32-esp/bin"
PREFIX = "riscv32-esp-elf-"

# .dram0.dummy is a placeholder the linker uses to mirror the flash rodata
# reservation into the DRAM address space. Counting it as "used DRAM" would
# overstate usage by tens of KB, so it is excluded from the used total and
# reported separately.
DRAM_USED = (".dram0.data", ".dram0.bss")
IRAM_USED = (".iram0.text", ".iram0.data", ".iram0.bss", ".iram0.text_end")
FLASH_USED = (".flash.text", ".flash.rodata", ".flash.appdesc")


def _repo_relative(path: str) -> str:
    """Record paths relative to the repository root.

    An absolute path bakes one machine's home directory into a committed
    artifact: it leaks a username, and it makes two captures of the same build
    on different machines diff as though they differed.
    """
    p = pathlib.Path(path).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            try:
                return str(p.relative_to(parent))
            except ValueError:
                break
    return p.name


def _tool(name: str, toolchain: str | None) -> str:
    base = pathlib.Path(toolchain) if toolchain else DEFAULT_TOOLCHAIN
    p = base / (PREFIX + name)
    return str(p) if p.exists() else (PREFIX + name)


def _run(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def sections(elf: str, toolchain: str | None = None) -> dict[str, int]:
    out = _run([_tool("size", toolchain), "-A", elf])
    got = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("."):
            try:
                got[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return got


def dram_capacity(mapfile: str) -> int | None:
    """Read the DRAM segment length from the link map rather than hard-coding it.

    The C3's usable DRAM is not simply "320 KB" - the ROM and the cache carve
    pieces out, and the exact figure moves with the IDF version. Parsing
    `dram0_0_seg` from the map is the difference between a capacity number that
    is true for this build and one that was true for some build, once.
    """
    try:
        text = pathlib.Path(mapfile).read_text(errors="replace")
    except OSError:
        return None
    m = re.search(r"dram0_0_seg\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)", text)
    return int(m.group(1), 16) if m else None


def top_bss(elf: str, limit: int = 15, toolchain: str | None = None) -> list[dict]:
    """Largest statically allocated objects, so a memory regression has a name.

    Cross-checked by the analyzer against the device's own arena census: if the
    firmware claims six buffers totalling 240 KB and the ELF disagrees, one of
    the two is stale and the artifact says so.
    """
    try:
        out = _run([_tool("nm", toolchain), "--size-sort", "-S", "-td", "-C", elf])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        _addr, size, kind, name = parts
        if kind.lower() not in ("b", "d"):   # bss / initialised data
            continue
        try:
            rows.append({"symbol": name.strip(), "bytes": int(size),
                         "kind": "bss" if kind.lower() == "b" else "data"})
        except ValueError:
            continue
    rows.sort(key=lambda r: -r["bytes"])
    return rows[:limit]


def embedded_weights(elf: str, toolchain: str | None = None) -> dict[str, int]:
    """Size of each embedded weight blob, from its linker-generated symbols.

    PlatformIO's `board_build.embed_files` brackets each blob with
    _binary_<name>_start/_end, so the size is the difference - which is how the
    2.4 MB of flash weights gets attributed to weights rather than to code.
    """
    try:
        out = _run([_tool("nm", toolchain), "-td", elf])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}
    addrs = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2].startswith("_binary_"):
            try:
                addrs[parts[2]] = int(parts[0])
            except ValueError:
                pass
    out_sizes = {}
    for sym, addr in addrs.items():
        if sym.endswith("_start"):
            end = sym[: -len("_start")] + "_end"
            if end in addrs:
                out_sizes[sym[len("_binary_"): -len("_start")]] = addrs[end] - addr
    return out_sizes


def analyze(elf: str, mapfile: str | None = None,
            toolchain: str | None = None) -> dict:
    sec = sections(elf, toolchain)
    dram_used = sum(sec.get(s, 0) for s in DRAM_USED)
    iram_used = sum(sec.get(s, 0) for s in IRAM_USED)
    flash_used = sum(sec.get(s, 0) for s in FLASH_USED)
    cap = dram_capacity(mapfile) if mapfile else None
    weights = embedded_weights(elf, toolchain)

    return {
        "elf": _repo_relative(elf),
        "sections": sec,
        "dram": {
            "data": sec.get(".dram0.data", 0),
            "bss": sec.get(".dram0.bss", 0),
            "used": dram_used,
            "capacity": cap,
            "free": (cap - dram_used) if cap else None,
            "dummy_excluded": sec.get(".dram0.dummy", 0),
            "capacity_source": "firmware.map dram0_0_seg" if cap else "unknown",
        },
        "iram": {"used": iram_used},
        "flash": {
            "text": sec.get(".flash.text", 0),
            "rodata": sec.get(".flash.rodata", 0),
            "used": flash_used,
            "embedded_weights": weights,
            "embedded_weights_total": sum(weights.values()),
            "code_and_data_excl_weights": flash_used - sum(weights.values()),
        },
        "top_static": top_bss(elf, toolchain=toolchain),
        "measured": True,
        "method": "riscv32-esp-elf-size -A and -nm on the linked ELF; DRAM capacity from the link map",
    }


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("elf")
    ap.add_argument("--map", dest="mapfile")
    ap.add_argument("--toolchain")
    a = ap.parse_args()
    print(json.dumps(analyze(a.elf, a.mapfile, a.toolchain), indent=1))
