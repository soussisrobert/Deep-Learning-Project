"""Round 5 queues: failure / attack awareness (gated) and natural corruptions incl. synthetic night.

  python plan/round5/make_plan5.py   # writes plan/round5/queue/*.tsv, plan/round5/SCHEDULE.md, plan/merges.json

Submit only what scripts/r5_gate.py lists under NEXT. See ROUND5_PLAN.md.
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
NT83 = json.loads((ROOT / "plan/round5/nontest-83.json").read_text())["sequences"]

MS = {"clean": 18.6, "pgd10": 164.0, "shift10": 170.0,   # measured rounds 1-4 (shift: estimate, +1 max per step)
      "corrupt": 23.0, "corrupt_def": 25.0, "cpu_int8": 60.0}
BASE = "--dataset flat --root data/uav123_flat --snapshot weights/general_model.pth --out results"
TEST, DEVF = "--seqs-file attack-subset-40.json", "--seqs-file plan/round3/dev-20.json"
T, D = "uav123-sub40_", "uav123-dev20_"
G = "--device cuda"
A10 = "--attack pgd --eps 4 --steps 10"
LAMBDAS = (1, 3, 10)
LSEL = "$R5L"
PRE = "R5L=$(python scripts/r5_selected.py) && "
INT8 = "--device cpu --threads 4 --variant int8bb-s1 --calib-seqs 40 --calib-frames 4 --calib-spread --calib-set 1"
CORR = ["gauss_noise:3", "gauss_blur:3", "motion_blur:3", "contrast:3", "low_light:3", "low_light:5", "night:3"]


def steps():
    s = {}
    # ---- F/A pilot on DEV (all runs save per-frame confidence: suffix _sc)
    s["P0"] = [(D + "fp32_clean_nodef_sc", G, DEV, MS["clean"], "gpu"),
               (D + "fp32_pgd4s10_nodef_sc", f"{G} {A10}", DEV, MS["pgd10"], "gpu")]
    s["P1"] = [(D + f"fp32_pgd4s10shift{l}_nodef_sc", f"{G} {A10} --objective shift --shift-lambda {l}", DEV, MS["shift10"], "gpu")
               for l in LAMBDAS]
    # ---- TEST
    s["F1"] = [(T + "fp32_clean_nodef_sc", G, S40, MS["clean"], "gpu")]
    s["A1"] = [(T + "fp32_pgd4s10_nodef_sc", f"{G} {A10}", S40, MS["pgd10"], "gpu")]
    s["A2"] = [(T + "fp32_pgd4s10shiftsel_nodef_sc", f"{G} {A10} --objective shift --shift-lambda {LSEL}", S40, MS["shift10"], "gpu")]
    # ---- K: corruptions (no gate; descriptive + int8 parity)
    k1 = []
    for c in CORR:
        tag = c.replace(":", "")
        k1.append((T + f"fp32_{tag}_nodef_sc", f"{G} --corrupt {c}", S40, MS["corrupt"], "gpu"))
        k1.append((T + f"fp32_{tag}_jpeg75_sc", f"{G} --corrupt {c} --defense jpeg75", S40, MS["corrupt_def"], "gpu"))
    s["K1"] = k1
    s["K2"] = [(T + f"int8s1_{c.replace(':', '')}_t4_sc", f"{INT8} --corrupt {c}", S40, MS["cpu_int8"], "cpu") for c in CORR]
    # ---- R4C: round-4 confirmation (no gate; CPU only)
    s["R4C1"] = [("bench/r4c_attributes", "python scripts/export_attributes.py && mkdir -p results/bench/r4c_attributes "
                  "&& echo '{}' > results/bench/r4c_attributes/summary.json", None, 0.02, "cpu")]
    # selected recipe on the 83 non-TEST videos; calibration from the 40 TEST videos (roles swapped, never overlapping).
    # cal0 = exact recipe (all 40 x 4 spread frames); cal1/cal2 = sensitivity draws with 30 of the 40 videos.
    nt = "--seqs-file plan/round5/nontest-83.json"
    s["R4C2"] = [("uav123-nt83_int8sel_clean_nodef_t4_cal0", f"{INT8.replace('--calib-set 1', '--calib-set 0')}", NT83, MS["cpu_int8"], "cpu"),
                 ("uav123-nt83_int8sel30_clean_nodef_t4_cal1", f"{INT8.replace('--calib-seqs 40', '--calib-seqs 30')}", NT83, MS["cpu_int8"], "cpu"),
                 ("uav123-nt83_int8sel30_clean_nodef_t4_cal2",
                  f"{INT8.replace('--calib-seqs 40', '--calib-seqs 30').replace('--calib-set 1', '--calib-set 2')}", NT83, MS["cpu_int8"], "cpu")]
    return s


def cmd(name, sf, args):
    c = f"python scripts/run_track.py {BASE} {sf} {args} --name {name}"
    return (PRE + c) if LSEL in args else c


def main():
    qdir = ROOT / "plan/round5/queue"
    qdir.mkdir(parents=True, exist_ok=True)
    for f in qdir.glob("*.tsv"):
        f.unlink()
    merges, rows = [], []
    for step, conds in steps().items():
        units = []
        for name, args, seqs, ms, kind in conds:
            if seqs is None:                      # a plain command (attribute export)
                units.append((name, args, ms, kind))
                continue
            sf = TEST if seqs is S40 else (DEVF if seqs is DEV else "--seqs-file plan/round5/nontest-83.json")
            parts = split(seqs, ms)
            if len(parts) == 1:
                units.append((name, cmd(name, sf, args), hours(seqs, ms), kind))
                continue
            names = []
            for i, p in enumerate(parts, 1):
                pf = f"plan/round5/parts/{name}__p{i}.txt"
                (ROOT / pf).parent.mkdir(parents=True, exist_ok=True)
                (ROOT / pf).write_text("\n".join(p) + "\n")
                names.append(f"{name}__p{i}")
                units.append((names[-1], cmd(names[-1], f"--seqs-file {pf}", args), hours(p, ms), kind))
            merges.append({"name": name, "parts": names, "expect": "attack-subset-40.json" if seqs is S40 else None})
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
            q = f"plan/round5/queue/{step}{'' if len(jobs) == 1 else '-' + str(j)}_{us[0][3]}.tsv"
            (ROOT / q).write_text("".join(f"{u[0]}\t{u[1]}\n" for u in us))
            rows.append((step, us[0][3], h, q, [u[0] for u in us]))
    (ROOT / "plan/merges.json").write_text(json.dumps(merges, indent=1))
    md = ["# Round 5 schedule (generated by plan/round5/make_plan5.py; hours are estimates from measured rates)", "",
          "Submit a queue **only when `scripts/r5_gate.py` lists it under NEXT**. GPU queues: `slurm/job_queue_gpu_tc2.sbatch`; "
          "cpu queues: `slurm/job_queue_cpu_tc2.sbatch`.", "",
          "| Step | Queue file | Needs | Est. hours | Runs |", "|---|---|---|---|---|"]
    for step, kind, h, q, names in rows:
        need = {"gpu": "GPU", "cpu": "CPU (4 threads)"}[kind]
        md.append(f"| {step} | `{q}` | {need} | {h:.1f} | " + ", ".join(n.replace(T, '').replace(D, 'dev:') for n in names) + " |")
    md += ["", f"Everything, if no branch is cut: GPU {sum(r[2] for r in rows if r[1] == 'gpu'):.1f} h, "
           f"CPU {sum(r[2] for r in rows if r[1] == 'cpu'):.1f} h (CPU runs beside GPU jobs).",
           "", "Split conditions (merged automatically): " + (", ".join(m["name"] for m in merges) or "none")]
    (ROOT / "plan/round5/SCHEDULE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
