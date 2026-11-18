"""Build the next-round job queues for TC2 from measured run times.

  python plan/make_plan.py            # writes plan/queue/*.tsv and plan/SCHEDULE.md

Every experiment is one tracking run (one condition). A condition that would not fit a
job's time budget is split into sequence parts (<name>__p1, __p2 ...) and merged later by
scripts/merge_all.py. Queue files are plain text: one run per line, <run name> TAB <command>.
slurm/job_queue_tc2.sbatch runs a queue, skipping runs whose summary.json already exists.

Cost model (milliseconds per frame, whole job including image loading). Measured values
come from the 16-17 Oct TC2 runs (A40 GPU; Xeon Gold 6230R CPU); estimates are marked.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FR = json.loads((ROOT / "plan/uav123_frames.json").read_text())["frames"]
S40 = json.loads((ROOT / "attack-subset-40.json").read_text())["sequences"]
ALL = sorted(FR)
BUDGET_H = 5.2          # of the 6 h wall; leaves room for start-up, slow nodes and merges

MS = {  # ms per frame, whole job
    ("gpu", "clean"): 18.6,      # measured: 11.7 min / 37,681 frames
    ("gpu", "noise"): 19.3,      # measured
    ("gpu", "fgsm"): 35.2,       # measured
    ("gpu", "pgd"): 164.0,       # measured, 10 steps
    ("gpu", "def_clean"): 21.0,  # estimate: clean + input transform
    ("gpu", "def_pgd"): 167.0,   # estimate: PGD + one transform per frame
    ("gpu", "def_adapt"): 190.0,  # estimate: PGD through the transform at every step (+15%)
    ("cpu", "fp32"): 100.0,      # measured: 187.5 min / 112,578 frames, 4 threads
    ("cpu", "int8bb"): 60.0,     # estimate from network latency 43 vs 89 ms
    ("cpu", "int8full"): 55.0,   # estimate (head convs int8 as well)
    ("cpu", "prune50"): 100.0,   # estimate: no speed-up measured
    ("mix", "int8_pgd"): 220.0,  # estimate: int8 tracking on CPU + PGD computed on the GPU
}
DEFENCES = ["blur1", "median3", "bits5", "jpeg75"]
BASE = "--dataset flat --root data/uav123_flat --snapshot weights/general_model.pth --out results"
SUB = "--seqs-file attack-subset-40.json"


def hours(seqs, ms):
    return sum(FR[s] for s in seqs) * ms / 3.6e6


def split(seqs, ms):
    """Balanced parts so each fits the budget (greedy by frames)."""
    k = 1
    while hours(seqs, ms) / k > BUDGET_H:
        k += 1
    parts = [[] for _ in range(k)]
    for s in sorted(seqs, key=lambda s: -FR[s]):
        min(parts, key=lambda p: sum(FR[x] for x in p)).append(s)
    return [sorted(p) for p in parts]


def conditions():
    """(priority, kind, name, args, sequences, ms/frame, purpose)"""
    c = []
    # P1: defences on the attack subset (the proposed improvement)
    for d in DEFENCES:
        c.append((1, "gpu", f"uav123-sub40_fp32_clean_{d}", f"--device cuda --defense {d}", S40, MS["gpu", "def_clean"],
                  "clean cost of the defence"))
        c.append((1, "gpu", f"uav123-sub40_fp32_pgd4s10_{d}", f"--device cuda --attack pgd --eps 4 --steps 10 --defense {d}",
                  S40, MS["gpu", "def_pgd"], "defence vs a defence-unaware attacker"))
        c.append((1, "gpu", f"uav123-sub40_fp32_pgd4s10_{d}_adaptive",
                  f"--device cuda --attack pgd --eps 4 --steps 10 --defense {d} --adaptive",
                  S40, MS["gpu", "def_adapt"], "defence vs an attacker who knows it (the honest number)"))
    # P2: compression accuracy on CPU, same 40 sequences (fp32 CPU reference already exists for all 123)
    for v in ["int8bb", "int8full", "prune50"]:
        c.append((2, "cpu", f"uav123-sub40_{v}_clean_nodef_t4", f"--device cpu --threads 4 --variant {v}", S40,
                  MS["cpu", v], "accuracy cost of compression"))
    # P3: complete the attack curve, check seed stability
    c.append((3, "gpu", "uav123-sub40_fp32_noise1_nodef", "--device cuda --attack noise --eps 1", S40, MS["gpu", "noise"],
              "control for FGSM eps 1"))
    for e in (1, 2, 8):
        c.append((3, "gpu", f"uav123-sub40_fp32_pgd{e}s10_nodef", f"--device cuda --attack pgd --eps {e} --steps 10", S40,
                  MS["gpu", "pgd"], "PGD budget curve"))
    for sd in (1, 2):
        c.append((3, "gpu", f"uav123-sub40_fp32_pgd4s10_nodef_seed{sd}",
                  f"--device cuda --attack pgd --eps 4 --steps 10 --seed {sd}", S40, MS["gpu", "pgd"], "seed stability"))
        c.append((3, "gpu", f"uav123-sub40_fp32_noise4_nodef_seed{sd}", f"--device cuda --attack noise --eps 4 --seed {sd}",
                  S40, MS["gpu", "noise"], "seed stability of the control"))
    # P4: does compression change robustness? attacker uses the fp32 weights (white-box surrogate)
    for v in ["int8bb", "int8full"]:
        c.append((4, "mix", f"uav123-sub40_{v}_pgd4s10_nodef_t4",
                  f"--device cpu --threads 4 --variant {v} --attack pgd --eps 4 --steps 10 --attack-device cuda",
                  S40, MS["mix", "int8_pgd"], "robustness after compression"))
    # P5: full-benchmark accuracy of the compressed models (paper-comparable, 123 sequences)
    for v in ["int8bb", "int8full"]:
        c.append((5, "cpu", f"uav123_{v}_clean_nodef_t4", f"--device cpu --threads 4 --variant {v}", ALL, MS["cpu", v],
                  "full UAV123 accuracy of the compressed model"))
    return c


def main():
    qdir = ROOT / "plan/queue"
    qdir.mkdir(parents=True, exist_ok=True)
    for f in qdir.glob("*.tsv"):
        f.unlink()
    units, merges = [], []
    for pri, kind, name, args, seqs, ms, why in conditions():
        parts = split(seqs, ms)
        if len(parts) == 1:
            units.append((pri, kind, name, f"python scripts/run_track.py {BASE} {SUB if seqs is S40 else ''} {args} --name {name}",
                          hours(seqs, ms), why))
        else:
            names = []
            for i, p in enumerate(parts, 1):
                pf = f"plan/parts/{name}__p{i}.txt"
                (ROOT / pf).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / pf).write_text("\n".join(p) + "\n")
                n = f"{name}__p{i}"
                names.append(n)
                units.append((pri, kind, n, f"python scripts/run_track.py {BASE} --seqs-file {pf} {args} --name {n}",
                              hours(p, ms), why + f" (part {i}/{len(parts)})"))
            merges.append({"name": name, "parts": names,
                           "expect": "attack-subset-40.json" if seqs is S40 else None})
    (ROOT / "plan/merges.json").write_text(json.dumps(merges, indent=1))
    # pack into jobs: gpu and mix need the GPU, cpu does not; keep priority order
    jobs = []
    for kind in ("gpu", "mix", "cpu"):
        cur, h = [], 0.0
        for u in sorted([u for u in units if u[1] == kind], key=lambda u: u[0]):
            if cur and (h + u[4] > BUDGET_H or u[0] != cur[-1][0]):
                jobs.append((kind, cur, h))
                cur, h = [], 0.0
            cur.append(u)
            h += u[4]
        if cur:
            jobs.append((kind, cur, h))
    jobs.sort(key=lambda j: (min(u[0] for u in j[1]), j[0] != "gpu"))
    rows = []
    for i, (kind, us, h) in enumerate(jobs, 1):
        q = f"plan/queue/{i:02d}_{kind}_P{us[0][0]}.tsv"
        (ROOT / q).write_text("".join(f"{u[2]}\t{u[3]}\n" for u in us))
        rows.append((i, kind, us[0][0], h, q, [u[2] for u in us], {u[5] for u in us}))
    tot = {k: sum(r[3] for r in rows if r[1] == k) for k in ("gpu", "mix", "cpu")}
    md = ["# Next-round schedule (generated by plan/make_plan.py; times are estimates)", "",
          f"Budget per job {BUDGET_H} h of the 6 h wall. GPU work {tot['gpu']:.1f} h, GPU+CPU work {tot['mix']:.1f} h, "
          f"CPU work {tot['cpu']:.1f} h. TC2 runs two jobs at once: pair one GPU job with one CPU job.", "",
          "Submit each from the login node in ~/hift-stress:", "",
          "```", "sbatch --export=ALL,QUEUE=<queue file> slurm/job_queue_gpu_tc2.sbatch   # gpu and mix queues",
          "sbatch --export=ALL,QUEUE=<queue file> slurm/job_queue_cpu_tc2.sbatch   # cpu queues", "```", "",
          "| Job | Needs | Priority | Est. hours | Queue file | Runs | Purpose |", "|---|---|---|---|---|---|---|"]
    for i, kind, pri, h, q, names, why in rows:
        need = {"gpu": "GPU", "mix": "GPU + 4 CPU", "cpu": "CPU (4 threads)"}[kind]
        md.append(f"| {i} | {need} | P{pri} | {h:.1f} | `{q}` | {len(names)}: " + ", ".join(n.replace('uav123-sub40_', '') for n in names)
                  + f" | {'; '.join(sorted(why))} |")
    md += ["", f"Split conditions (merged automatically at the end of every queue job by `scripts/merge_all.py`): "
           + (", ".join(m["name"] for m in merges) or "none")]
    (ROOT / "plan/SCHEDULE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
