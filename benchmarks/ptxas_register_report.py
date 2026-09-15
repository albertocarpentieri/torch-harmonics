"""
Registers per thread for each instantiation of the ragged attention kernels.

Occupancy on this kernel is register-limited, so the NB (neighbours in flight) and
NLOC (accumulator registers) constants trade instruction-level parallelism against
how many warps an SM can hold. That trade can be read straight off ptxas -v without
a GPU, which is the only reason this exists: it makes the constants tunable while the
scheduler is down.

Reads a ptxas verbose log on stdin or as argv[1] and prints, per instantiation, the
register count and the warps per SM it permits. Spills are called out separately
because a spilling kernel has already lost the argument -- the point of register
blocking is to keep the accumulator off memory.
"""

import re
import shutil
import subprocess
import sys

# Blackwell keeps 64K 32-bit registers per SM, allocated per warp of 32 lanes.
REGS_PER_SM = 65536
LANES = 32
# An SM tops out at 64 resident warps whatever the register budget allows.
MAX_WARPS_PER_SM = 64


def demangle(names):
    """Batch through c++filt; fall back to the mangled name if it is unavailable."""
    if not shutil.which("c++filt"):
        return {n: n for n in names}
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True).stdout
    return dict(zip(names, out.splitlines()))


def parse(log):
    blocks = re.findall(
        r"Compiling entry function '(\S+?)' for '(sm_\w+)'(.*?)(?=Compiling entry function|\Z)", log, re.S
    )
    names = demangle([m for m, _, _ in blocks])

    rows = []
    for mangled, arch, body in blocks:
        regs = re.search(r"Used (\d+) registers", body)
        if not regs:
            continue
        spill_st = re.search(r"(\d+) bytes spill stores", body)
        smem = re.search(r"(\d+) bytes smem", body)
        rows.append(
            {
                "name": names.get(mangled, mangled),
                "arch": arch,
                "regs": int(regs.group(1)),
                "spill": int(spill_st.group(1)) if spill_st else 0,
                "smem": int(smem.group(1)) if smem else 0,
            }
        )
    return rows


def kernel_label(name):
    """Short name plus template arguments, which are what the sweep varies."""
    short = "special" if "ragged_special" in name else "generic" if "ragged_generic" in name else None
    if short is None:
        return None
    targs = re.search(r"<(.*)>", name)
    if not targs:
        return short
    args = [a.strip() for a in targs.group(1).split(",")]
    # <BDIM_X, BDIM_Y, NLOC, STORAGE_T> for special, <BDIM_X, STORAGE_T> for generic
    if short == "special" and len(args) >= 4:
        return f"special  NLOC={args[2]:>2}  {args[3]}"
    return f"{short}  {args[-1]}"


def main():
    log = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    rows = [r for r in parse(log) if kernel_label(r["name"])]
    if not rows:
        print("no ragged attention instantiations found in the log")
        return 1

    hdr = f"{'kernel':<44} {'regs':>5} {'spill':>6} {'smem':>6} {'warps/SM':>9} {'occupancy':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (kernel_label(r["name"]))):
        warps = min(MAX_WARPS_PER_SM, REGS_PER_SM // (r["regs"] * LANES)) if r["regs"] else MAX_WARPS_PER_SM
        flag = "  <-- SPILLING" if r["spill"] else ""
        print(
            f"{kernel_label(r['name']):<44} {r['regs']:>5} {r['spill']:>6} {r['smem']:>6} "
            f"{warps:>9} {warps / MAX_WARPS_PER_SM:>9.0%}{flag}"
        )

    print()
    print("warps/SM is the register budget's cap, not the achieved occupancy; shared")
    print("memory and block size can bind first. What matters for the NB sweep is the")
    print("trend: if raising NB drops warps/SM faster than it adds work in flight, it")
    print("is buying latency hiding with one hand and spending it with the other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
