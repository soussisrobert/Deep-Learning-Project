"""Round 3 job queues: does the JPEG + int8 result hold up, and can it pass the rules?

  python plan/round3/make_plan3.py      # writes plan/round3/queue/*.tsv, plan/round3/SCHEDULE.md, plan/merges.json

Stage 1 (choose, then stress): JPEG quality sweep on 20 held-out DEV videos; stronger adaptive
attacks on the round-2 JPEG75 defence on the fixed TEST 40.
--> scripts/r3_select.py picks the quality from DEV only (rule in ROUND3_PLAN.md) <--
Stage 2 (confirm): the selected quality ("sel") on TEST: clean, three adaptive attacks, seeds,
GPU-side JPEG, int8 + JPEG on CPU, natural corruptions.

Stage-2 commands read the choice at run time: --defense $(python scripts/r3_selected.py jpeg).
Cost model: ms per frame for the whole job, measured on the A40 / Xeon in round 2 except where
marked as an estimate (50-step and surrogate-gradient attacks scale the measured 10-step cost).
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mp", ROOT / "plan/make_plan.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)
FR, S40, hours, split = mp.FR, mp.S40, mp.hours, mp.split
DEV = json.loads((ROOT / "plan/round3/dev-20.json").read_text())["sequences"]
BUDGET_H = mp.BUDGET_H

MS = {
    "clean": 18.6, "def_clean": 21.0, "corrupt": 23.0,               # measured / measured+transform
    "pgd10": 164.0, "adapt10": 176.0,                              # measured (round 2 jpeg75 adaptive: 168 ms + load)
    "adapt10_diff": 195.0,                                         # estimate: + surrogate JPEG fwd/bwd per step
    "pgd50": 164.0 + 4 * 145.0,                                    # estimate: 5x the attack part of pgd10
    "adapt50": 176.0 + 4 * 155.0, "adapt50_diff": 195.0 + 4 * 175.0,
    "cpu_int8": 60.0, "cpu_int8_def": 62.0,                        # measured round 2 (+ 1.1 ms JPEG, bench)
    "mix_adapt10": 235.0, "mix_adapt10_diff": 255.0,               # estimate: round-2 mix 220 + defence
}
BASE = "--dataset flat --root data/uav123_flat --snapshot weights/general_model.pth --out results"
TEST, DEVF = "--seqs-file attack-subset-40.json", "--seqs-file plan/round3/dev-20.json"
T, D = "uav123-sub40_", "uav123-dev20_"
SEL = "$(python scripts/r3_selected.py jpeg)"      # e.g. jpeg90
TSEL = "$(python scripts/r3_selected.py tjpeg)"    # e.g. tjpeg90
A10 = "--attack pgd --eps 4 --steps 10"
A50 = "--attack pgd --eps 4 --steps 50"
CORR = ["gauss_noise:3", "gauss_blur:3", "motion_blur:3", "low_light:3", "contrast:3"]


def conditions():
    """(stage, priority, kind, name, args, seqs, ms, purpose)"""
    c = []
    g = "--device cuda"
    # ---------------- stage 1: DEV sweep (choose the quality) ----------------
    c.append((1, 1, "gpu", D + "fp32_clean_nodef", f"{g}", DEV, MS["clean"], "DEV reference"))
    c.append((1, 1, "gpu", D + "fp32_pgd4s10_nodef", f"{g} {A10}", DEV, MS["pgd10"], "DEV attack reference"))
    for q in (75, 85, 90, 95):
        c.append((1, 1, "gpu", D + f"fp32_clean_jpeg{q}", f"{g} --defense jpeg{q}", DEV, MS["def_clean"], "DEV clean cost per quality"))
        c.append((1, 1, "gpu", D + f"fp32_pgd4s10_jpeg{q}_adaptive_diff", f"{g} {A10} --defense jpeg{q} --adaptive --grad diff",
                  DEV, MS["adapt10_diff"], "DEV robustness per quality (selection attack)"))
    # defence timing next to the network (rule H8), tiny jobs
    bd = "python scripts/bench_defense.py --dataset flat --root data/uav123_flat --snapshot weights/general_model.pth"
    c.append((1, 1, "gpu", "bench/defense_cuda_fp32", f"BENCH {bd} --device cuda --out results/bench/defense_cuda_fp32",
              None, 0.05, "defence time on the GPU"))
    for v, t in (("fp32", 4), ("int8bb", 4), ("int8bb", 1)):
        c.append((1, 1, "cpu", f"bench/defense_cpu_{v}_t{t}",
                  f"BENCH {bd} --device cpu --threads {t} --variant {v} --out results/bench/defense_cpu_{v}_t{t}",
                  None, 0.05, "defence time on the CPU"))
    # ---------------- stage 1: stronger attacks on round-2 JPEG75 (TEST) ----------------
    c.append((1, 2, "gpu", T + "fp32_pgd4s50_nodef", f"{g} {A50}", S40, MS["pgd50"], "stronger undefended baseline"))
    c.append((1, 2, "gpu", T + "fp32_pgd4s10_jpeg75_adaptive_diff", f"{g} {A10} --defense jpeg75 --adaptive --grad diff",
              S40, MS["adapt10_diff"], "JPEG75 vs surrogate-gradient attacker"))
    c.append((1, 2, "gpu", T + "fp32_pgd4s50_jpeg75_adaptive_diff", f"{g} {A50} --defense jpeg75 --adaptive --grad diff",
              S40, MS["adapt50_diff"], "JPEG75 vs strongest attacker (50 steps, surrogate)"))
    # ---------------- stage 2: selected quality on TEST ----------------
    c.append((2, 3, "gpu", T + "fp32_clean_jpegsel", f"{g} --defense {SEL}", S40, MS["def_clean"], "clean cost (rule H7)"))
    c.append((2, 3, "gpu", T + "fp32_pgd4s10_jpegsel_adaptive", f"{g} {A10} --defense {SEL} --adaptive", S40, MS["adapt10"],
              "BPDA-10, comparable with round 2"))
    c.append((2, 3, "gpu", T + "fp32_pgd4s10_jpegsel_adaptive_diff", f"{g} {A10} --defense {SEL} --adaptive --grad diff",
              S40, MS["adapt10_diff"], "surrogate-10"))
    c.append((2, 3, "gpu", T + "fp32_pgd4s50_jpegsel_adaptive_diff", f"{g} {A50} --defense {SEL} --adaptive --grad diff",
              S40, MS["adapt50_diff"], "surrogate-50 (rule H6)"))
    c.append((2, 3, "gpu", T + "fp32_clean_tjpegsel", f"{g} --defense {TSEL}", S40, MS["def_clean"], "GPU JPEG: clean + time (H8)"))
    c.append((2, 3, "gpu", T + "fp32_pgd4s10_tjpegsel_adaptive_diff", f"{g} {A10} --defense {TSEL} --adaptive --grad diff",
              S40, MS["adapt10_diff"], "GPU JPEG robustness"))
    # CPU deployment: int8 calibration spread + int8 with JPEG
    for k in (1, 2, 3):
        c.append((2, 4, "cpu", T + f"int8bb_clean_nodef_t4_cal{k}", f"--device cpu --threads 4 --variant int8bb --calib-set {k}",
                  S40, MS["cpu_int8"], "int8 calibration spread (H9)"))
    c.append((2, 4, "cpu", T + "int8bb_clean_jpegsel_t4", f"--device cpu --threads 4 --variant int8bb --defense {SEL}",
              S40, MS["cpu_int8_def"], "combined clean + CPU overhead (H8-CPU, H10)"))
    c.append((2, 4, "mix", T + "int8bb_pgd4s10_jpegsel_adaptive_t4",
              f"--device cpu --threads 4 --variant int8bb {A10} --defense {SEL} --adaptive --attack-device cuda",
              S40, MS["mix_adapt10"], "combined vs BPDA-10 (H10)"))
    c.append((2, 4, "mix", T + "int8bb_pgd4s10_jpegsel_adaptive_diff_t4",
              f"--device cpu --threads 4 --variant int8bb {A10} --defense {SEL} --adaptive --grad diff --attack-device cuda",
              S40, MS["mix_adapt10_diff"], "combined vs surrogate-10 (H10)"))
    # natural corruptions (no attacker)
    for cs in CORR:
        tag = cs.replace(":", "")
        c.append((2, 5, "gpu", T + f"fp32_{tag}_nodef", f"{g} --corrupt {cs}", S40, MS["corrupt"], "natural corruption"))
        c.append((2, 5, "gpu", T + f"fp32_{tag}_jpegsel", f"{g} --corrupt {cs} --defense {SEL}", S40, MS["corrupt"],
                  "natural corruption with JPEG"))
    # lower priority: transparency and stability
    for q in (85, 90, 95):
        c.append((2, 6, "gpu", T + f"fp32_clean_jpeg{q}", f"{g} --defense jpeg{q}", S40, MS["def_clean"],
                  "clean cost of every quality on TEST (reported, not used to choose)"))
    for sd in (1, 2):
        c.append((2, 6, "gpu", T + f"fp32_pgd4s10_jpegsel_adaptive_diff_seed{sd}",
                  f"{g} {A10} --defense {SEL} --adaptive --grad diff --seed {sd}", S40, MS["adapt10_diff"], "seed stability"))
    c.append((1, 7, "gpu", T + "fp32_pgd4s50_jpeg75_adaptive", f"{g} {A50} --defense jpeg75 --adaptive",
              S40, MS["adapt50"], "JPEG75 vs 50-step BPDA (completeness)"))
    return c


def main():
    qdir = ROOT / "plan/round3/queue"
    qdir.mkdir(parents=True, exist_ok=True)
    for f in qdir.glob("*.tsv"):
        f.unlink()
    units, merges = [], []
    for stage, pri, kind, name, args, seqs, ms, why in conditions():
        if args.startswith("BENCH "):
            units.append((stage, pri, kind, name, args[6:], ms, why))
            continue
        sf = TEST if seqs is S40 else DEVF
        parts = split(seqs, ms)
        if len(parts) == 1:
            units.append((stage, pri, kind, name, f"python scripts/run_track.py {BASE} {sf} {args} --name {name}",
                          hours(seqs, ms), why))
            continue
        names = []
        for i, p in enumerate(parts, 1):
            pf = f"plan/round3/parts/{name}__p{i}.txt"
            (ROOT / pf).parent.mkdir(parents=True, exist_ok=True)
            (ROOT / pf).write_text("\n".join(p) + "\n")
            n = f"{name}__p{i}"
            names.append(n)
            units.append((stage, pri, kind, n, f"python scripts/run_track.py {BASE} --seqs-file {pf} {args} --name {n}",
                          hours(p, ms), why + f" (part {i}/{len(parts)})"))
        merges.append({"name": name, "parts": names, "expect": "attack-subset-40.json" if seqs is S40 else None})
    (ROOT / "plan/merges.json").write_text(json.dumps(merges, indent=1))
    jobs = []
    for kind in ("gpu", "mix", "cpu"):
        cur, h = [], 0.0
        for u in sorted([u for u in units if u[2] == kind], key=lambda u: (u[0], u[1])):
            if cur and (h + u[5] > BUDGET_H or u[0] != cur[-1][0] or u[1] != cur[-1][1]):
                jobs.append((kind, cur, h))
                cur, h = [], 0.0
            cur.append(u)
            h += u[5]
        if cur:
            jobs.append((kind, cur, h))
    jobs.sort(key=lambda j: (j[1][0][0], j[1][0][1], j[0] != "gpu"))
    rows = []
    for i, (kind, us, h) in enumerate(jobs, 1):
        q = f"plan/round3/queue/{i:02d}_S{us[0][0]}_{kind}_P{us[0][1]}.tsv"
        (ROOT / q).write_text("".join(f"{u[3]}\t{u[4]}\n" for u in us))
        rows.append((i, kind, us[0][0], us[0][1], h, q, [u[3] for u in us], sorted({u[6].split(' (part')[0] for u in us})))
    tot = {s: sum(r[4] for r in rows if r[2] == s) for s in (1, 2)}
    md = ["# Round 3 schedule (generated by plan/round3/make_plan3.py; hours are estimates)", "",
          f"Stage 1: {tot[1]:.1f} job-hours. Stage 2: {tot[2]:.1f} job-hours. Budget per job {BUDGET_H} h of the 6 h wall.",
          "TC2 runs two jobs at once: keep one GPU job running and use the second slot for CPU/mix jobs or the next GPU job.",
          "", "**Stage 2 starts only after `python scripts/r3_select.py` has written `plan/round3/selected.json`** "
          "(it needs the stage-1 DEV runs; priority P1 jobs).", "",
          "```", "sbatch --export=ALL,QUEUE=<queue file> slurm/job_queue_gpu_tc2.sbatch   # gpu and mix queues",
          "sbatch --export=ALL,QUEUE=<queue file> slurm/job_queue_cpu_tc2.sbatch   # cpu queues", "```", "",
          "| Job | Stage | Needs | Priority | Est. hours | Queue file | Runs | Purpose |", "|---|---|---|---|---|---|---|---|"]
    for i, kind, st, pri, h, q, names, why in rows:
        need = {"gpu": "GPU", "mix": "GPU + 4 CPU", "cpu": "CPU (4 threads)"}[kind]
        md.append(f"| {i} | {st} | {need} | P{pri} | {h:.1f} | `{q}` | {len(names)}: "
                  + ", ".join(n.replace(T, '').replace(D, 'dev:') for n in names) + f" | {'; '.join(why)} |")
    md += ["", "Split conditions (merged automatically at the end of every queue job): "
           + (", ".join(m["name"] for m in merges) or "none")]
    (ROOT / "plan/round3/SCHEDULE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
