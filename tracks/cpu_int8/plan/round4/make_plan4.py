"""Round 4 job queues (gated: losers are cut, winners continue). See ROUND4_PLAN.md.

  python plan/round4/make_plan4.py     # writes plan/round4/queue/*.tsv, plan/round4/SCHEDULE.md, plan/merges.json

Branches:
  J  JPEG-85 (best DEV trade-off in round 3) on TEST: clean -> surrogate-10 -> surrogate-50 + BPDA-10
  Q  int8 backbone: DEV pilot of 4 calibration recipes x 3 draws -> best recipe on TEST x 3 draws
  C  int8 (selected recipe) + JPEG-85 on CPU: clean -> attacks
Which queue to submit next is decided by scripts/r4_gate.py, never by hand.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("mp", ROOT / "plan/make_plan.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)
FR, S40, hours, split, BUDGET_H = mp.FR, mp.S40, mp.hours, mp.split, mp.BUDGET_H
DEV = json.loads((ROOT / "plan/round3/dev-20.json").read_text())["sequences"]

MS = {  # ms per frame, whole job; measured in rounds 2-3 on TC2 unless marked
    "def_clean": 21.0, "adapt10": 176.0, "adapt10_diff": 190.0,        # round 3 measured: 1h59m / 37,681 frames
    "adapt50_diff": 850.0,                                             # round 3 measured: 8h51m / 37,681 frames
    "cpu_fp32": 100.0, "cpu_fp32_def": 102.0, "cpu_int8": 60.0, "cpu_int8_s1": 80.0,  # s1: estimate (+20 ms)
    "cpu_int8_def": 62.0, "mix10": 240.0, "mix10_def": 250.0, "mix10_diff": 275.0,
}
BASE = "--dataset flat --root data/uav123_flat --snapshot weights/general_model.pth --out results"
TEST, DEVF = "--seqs-file attack-subset-40.json", "--seqs-file plan/round3/dev-20.json"
T, D = "uav123-sub40_", "uav123-dev20_"
A10, A50 = "--attack pgd --eps 4 --steps 10", "--attack pgd --eps 4 --steps 50"
G = "--device cuda"
CPU = "--device cpu --threads 4"
SEL = "$R4SEL"   # filled at run time from plan/round4/int8_selected.json (see cmd() below)
PRE = "R4SEL=$(python scripts/r4_selected.py) && "   # fails the run if the DEV choice has not been made yet


def cmd(name, sf, args):
    c = f"python scripts/run_track.py {BASE} {sf} {args} --name {name}"
    return (PRE + c) if SEL in args else c
RECIPES = {  # DEV pilot candidates (round-3 recipe first)
    "int8bb_c20x2": ("int8bb", "--calib-seqs 20 --calib-frames 2"),
    "int8bb_c40x4s": ("int8bb", "--calib-seqs 40 --calib-frames 4 --calib-spread"),
    "int8bb-s1_c20x2": ("int8bb-s1", "--calib-seqs 20 --calib-frames 2"),
    "int8bb-s1_c40x4s": ("int8bb-s1", "--calib-seqs 40 --calib-frames 4 --calib-spread"),
}


def steps():
    """step -> list of (name, args, seqs, ms, kind); kind gpu/cpu/mix"""
    s = {}
    # ---------- J: JPEG-85 on TEST
    s["J1"] = [(T + "fp32_clean_jpeg85", f"{G} --defense jpeg85", S40, MS["def_clean"], "gpu")]
    s["J1c"] = [(T + "fp32_clean_jpeg85_t4", f"{CPU} --defense jpeg85", S40, MS["cpu_fp32_def"], "cpu"),
                (T + "fp32_clean_nodef_t4", f"{CPU}", S40, MS["cpu_fp32"], "cpu")]
    s["J2"] = [(T + "fp32_pgd4s10_jpeg85_adaptive_diff", f"{G} {A10} --defense jpeg85 --adaptive --grad diff", S40,
                MS["adapt10_diff"], "gpu")]
    s["J3"] = [(T + "fp32_pgd4s50_jpeg85_adaptive_diff", f"{G} {A50} --defense jpeg85 --adaptive --grad diff", S40,
                MS["adapt50_diff"], "gpu"),
               (T + "fp32_pgd4s10_jpeg85_adaptive", f"{G} {A10} --defense jpeg85 --adaptive", S40, MS["adapt10"], "gpu")]
    # ---------- Q: int8 recipes, DEV pilot (TEST videos never used for calibration)
    q0 = [(D + "fp32_clean_nodef_t4", f"{CPU}", DEV, MS["cpu_fp32"], "cpu")]
    for tag, (var, cal) in RECIPES.items():
        for k in (1, 2, 3):
            q0.append((D + f"{tag}_cal{k}", f"{CPU} --variant {var} {cal} --calib-set {k} --calib-exclude attack-subset-40.json",
                       DEV, MS["cpu_int8_s1" if "s1" in var else "cpu_int8"], "cpu"))
    s["Q0"] = q0
    s["Q0b"] = [("bench/r4_edge_cpu",
                 "python scripts/bench_edge.py --snapshot weights/general_model.pth --dataset flat --root data/uav123_flat "
                 "--device cpu --variants fp32 int8bb int8bb-s1 --threads 4 1 --iters 200 --out results/edge_round4 "
                 "&& mkdir -p results/bench/r4_edge_cpu && echo '{}' > results/bench/r4_edge_cpu/summary.json",
                 None, 0.1, "cpu")]
    # ---------- Q1: selected recipe on TEST, 3 calibration draws (pool = all videos outside TEST)
    s["Q1"] = [(T + f"int8sel_clean_nodef_t4_cal{k}", f"{CPU} {SEL} --calib-set {k}", S40, MS["cpu_int8_s1"], "cpu")
               for k in (1, 2, 3)]
    # ---------- C: int8 (selected, draw 1) + JPEG-85
    s["C1"] = [(T + "int8sel_clean_jpeg85_t4", f"{CPU} {SEL} --calib-set 1 --defense jpeg85", S40, MS["cpu_int8_def"], "cpu")]
    s["C2"] = [(T + "int8sel_pgd4s10_nodef_t4", f"{CPU} {SEL} --calib-set 1 {A10} --attack-device cuda", S40, MS["mix10"], "mix"),
               (T + "int8sel_pgd4s10_jpeg85_adaptive_t4", f"{CPU} {SEL} --calib-set 1 {A10} --defense jpeg85 --adaptive "
                "--attack-device cuda", S40, MS["mix10_def"], "mix"),
               (T + "int8sel_pgd4s10_jpeg85_adaptive_diff_t4", f"{CPU} {SEL} --calib-set 1 {A10} --defense jpeg85 --adaptive "
                "--grad diff --attack-device cuda", S40, MS["mix10_diff"], "mix")]
    return s


def main():
    qdir = ROOT / "plan/round4/queue"
    qdir.mkdir(parents=True, exist_ok=True)
    for f in qdir.glob("*.tsv"):
        f.unlink()
    merges, rows = [], []
    for step, conds in steps().items():
        units = []
        for name, args, seqs, ms, kind in conds:
            if seqs is None:
                units.append((name, args, ms, kind))
                continue
            sf = TEST if seqs is S40 else DEVF
            parts = split(seqs, ms)
            if len(parts) == 1:
                units.append((name, cmd(name, sf, args), hours(seqs, ms), kind))
                continue
            names = []
            for i, p in enumerate(parts, 1):
                pf = f"plan/round4/parts/{name}__p{i}.txt"
                (ROOT / pf).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / pf).write_text("\n".join(p) + "\n")
                names.append(f"{name}__p{i}")
                units.append((names[-1], cmd(names[-1], f"--seqs-file {pf}", args),
                              hours(p, ms), kind))
            merges.append({"name": name, "parts": names, "expect": "attack-subset-40.json" if seqs is S40 else None})
        # pack each step into as few jobs as fit the budget
        jobs, cur, h = [], [], 0.0
        for u in units:
            if cur and (h + u[2] > BUDGET_H or u[3] != cur[-1][3]):
                jobs.append((cur, h))
                cur, h = [], 0.0
            cur.append(u)
            h += u[2]
        if cur:
            jobs.append((cur, h))
        for j, (us, h) in enumerate(jobs, 1):
            kind = us[0][3]
            q = f"plan/round4/queue/{step}{'' if len(jobs) == 1 else '-' + str(j)}_{kind}.tsv"
            (ROOT / q).write_text("".join(f"{u[0]}\t{u[1]}\n" for u in us))
            rows.append((step, kind, h, q, [u[0] for u in us]))
    (ROOT / "plan/merges.json").write_text(json.dumps(merges, indent=1))
    md = ["# Round 4 schedule (generated by plan/round4/make_plan4.py; hours are estimates from measured round-3 rates)", "",
          "Submit a queue **only when `scripts/r4_gate.py` lists it under NEXT**. GPU and mix queues use "
          "`slurm/job_queue_gpu_tc2.sbatch`; cpu queues use `slurm/job_queue_cpu_tc2.sbatch`.", "",
          "| Step | Queue file | Needs | Est. hours | Runs |", "|---|---|---|---|---|"]
    for step, kind, h, q, names in rows:
        need = {"gpu": "GPU", "mix": "GPU + 4 CPU", "cpu": "CPU (4 threads)"}[kind]
        md.append(f"| {step} | `{q}` | {need} | {h:.1f} | " + ", ".join(n.replace(T, '').replace(D, 'dev:') for n in names) + " |")
    tg = sum(r[2] for r in rows if r[1] != "cpu")
    tc = sum(r[2] for r in rows if r[1] == "cpu")
    md += ["", f"If every branch survives: GPU/mix {tg:.1f} h, CPU {tc:.1f} h (CPU jobs run next to GPU jobs). "
           "Branches cut by a gate are not run.",
           "", "Split conditions (merged automatically): " + (", ".join(m["name"] for m in merges) or "none")]
    (ROOT / "plan/round4/SCHEDULE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
