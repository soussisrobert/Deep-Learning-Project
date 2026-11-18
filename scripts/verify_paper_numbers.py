"""Level 1 reproduction: regenerate every number printed in the paper and check it.

Needs only Python + numpy + pandas (no GPU, no datasets). Run from the repo root:

    python scripts/verify_paper_numbers.py            # check against paper/expected_numbers.json
    python scripts/verify_paper_numbers.py --write    # also write results/numbers.json

Each check is one of:
  computed  - recomputed here from the released per-video / per-stretch files
  recorded  - read from a released measurement file (timing logs, frame-level analysis
              outputs); recomputing these needs raw tracker outputs (level 2/3, REPRODUCE.md)
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.stats import cluster_bootstrap, two_level_bootstrap, video_bootstrap  # noqa: E402

RES = ROOT / "results"
SPL = ROOT / "data/splits"
OUT: dict[str, dict] = {}


def put(key, value, kind="computed", source=""):
    OUT[key] = {"value": value, "kind": kind, "source": source}


def put_interval(key, triple, source):
    m, lo, hi = triple
    put(key + ".delta", m, source=source)
    put(key + ".lo", lo, source=source)
    put(key + ".hi", hi, source=source)


# ---------------------------------------------------------------- night
def night():
    def piv(name):
        d = pd.read_csv(RES / f"night/per_video_{name}.csv")
        return d.pivot(index="video", columns="model", values="success_auc"), d

    sets = {}
    for name in ["NAT2021", "NAT2021L", "UAVDark135_test", "UAVDark135_val"]:
        sets[name], raw = piv(name)
        assert sets[name].notna().all().all(), name
    # pooled-39: test rows first, then validation rows (order fixes the bootstrap draws)
    sets["UAVDark135_pooled39"] = pd.concat([sets["UAVDark135_test"], sets["UAVDark135_val"]])
    for name, p in sets.items():
        src = f"results/night/per_video_{name}.csv" if "pooled" not in name else "results/night/per_video_UAVDark135_{test,val}.csv"
        put(f"night.{name}.n", len(p), source=src)
        for m in ["original", "gamma", "gamma_anchor"]:
            put(f"night.{name}.{m}", float(p[m].mean()), source=src)
        for a, b in [("gamma", "original"), ("gamma_anchor", "original"), ("gamma_anchor", "gamma")]:
            put_interval(f"night.{name}.{a}-{b}", video_bootstrap((p[a] - p[b]).values, seed=20261118), src)
    d = pd.read_csv(RES / "night/per_video_NAT2021.csv")
    put("night.NAT2021.original.precision20", float(d[d.model == "original"].precision20.mean()),
        source="results/night/per_video_NAT2021.csv")
    d = pd.read_csv(RES / "night/per_video_NAT2021L.csv")
    put("night.NAT2021L.original.precision20", float(d[d.model == "original"].precision20.mean()),
        source="results/night/per_video_NAT2021L.csv")

    # E15 loss stretches (NAT2021, gamma+anchor)
    src = ROOT / "tracks/night/evidence/sources"
    st = [r for r in csv.DictReader(open(src / "E15_stretches.csv")) if r["dataset"] == "NAT2021"]
    put("night.E15.stretches", len(st), source="tracks/night/evidence/sources/E15_stretches.csv")
    put("night.E15.start_in_window", sum(r["gt_in_window_first_frame"] in ("1", "True", "true") for r in st),
        source="tracks/night/evidence/sources/E15_stretches.csv")
    h = [r for r in csv.DictReader(open(src / "E15_stretch_length_hist.csv")) if r["dataset"] == "NAT2021"]
    lost = sum(int(r["lost_frames"]) for r in h)
    long_ = sum(int(r["lost_frames"]) for r in h if r["length"] in ("101-200", "201+"))
    put("night.E15.lost_frames", lost, source="tracks/night/evidence/sources/E15_stretch_length_hist.csv")
    put("night.E15.lost_frames_in_101plus", long_, source="tracks/night/evidence/sources/E15_stretch_length_hist.csv")
    put("night.E15.share_101plus", long_ / lost, source="tracks/night/evidence/sources/E15_stretch_length_hist.csv")

    # overlap screen and split
    txt = (src / "output_HiFT_uavdark_leak_49103.out").read_text()
    m = re.search(r"frames matching NAT: (\d+)/(\d+)", txt)
    put("night.leak.matching_frames", int(m.group(1)), "recorded", "tracks/night/evidence/sources/output_HiFT_uavdark_leak_49103.out")
    put("night.leak.query_frames", int(m.group(2)), "recorded", "tracks/night/evidence/sources/output_HiFT_uavdark_leak_49103.out")
    put("night.leak.linked_pairs", int(re.search(r"(\d+) sequence pairs share footage", txt).group(1)), "recorded",
        "tracks/night/evidence/sources/output_HiFT_uavdark_leak_49103.out")
    split = {s: (SPL / f"night/UAVDark135_{s}.txt").read_text().split() for s in ["train", "val", "test"]}
    where = {v: s for s, vs in split.items() for v in vs}
    straddle = 0
    for r in csv.DictReader(open(SPL / "night/UAVDark135_groups.csv")):
        members = [x.strip().strip('"') for x in re.split(r"[;, ]+", r["group_members"]) if x.strip()]
        if {where.get(x) for x in members} != {r["split"]}:
            straddle += 1
    put("night.split.train", len(split["train"]), source="data/splits/night")
    put("night.split.val", len(split["val"]), source="data/splits/night")
    put("night.split.test", len(split["test"]), source="data/splits/night")
    put("night.split.groups_straddling", straddle, source="data/splits/night/UAVDark135_groups.csv")

    # speed (job 55514)
    for f in sorted(src.glob("speed_*.log")):
        times = [float(x) for x in re.findall(r"Time:\s*([\d.]+)s", f.read_text())]
        put(f"night.speed.{f.stem}.seconds", sum(times), "recorded", f"tracks/night/evidence/sources/{f.name}")


# ---------------------------------------------------------------- CPU
def cpu():
    d = pd.read_csv(RES / "cpu/per_video_UAV123.csv")
    src = "results/cpu/per_video_UAV123.csv"

    def p(m, s=0):
        return d[(d.model == m) & (d.seed == s)].set_index("video").success_auc

    s40 = sorted(json.loads((SPL / "cpu/uav123_test40.json").read_text())["sequences"])
    fp = p("fp32_cpu")
    draws = [p("int8bb-s1", k) for k in (1, 2, 3)]
    nt = p("int8bb-s1_nontest")
    D = np.stack([x[s40].values for x in draws])
    put("cpu.test40.fp32", float(fp[s40].mean()), source=src)
    put("cpu.test40.int8", float(D.mean()), source=src)
    put_interval("cpu.test40.two_level", two_level_bootstrap(D, fp[s40].values, seed=1), src)
    ch = {v: float(np.mean([x[v] for x in draws]) - fp[v]) for v in s40}
    ch.update({v: float(nt[v] - fp[v]) for v in nt.index})
    i8 = {v: float(np.mean([x[v] for x in draws])) for v in s40}
    i8.update({v: float(nt[v]) for v in nt.index})
    put("cpu.all123.n", len(ch), source=src)
    put("cpu.all123.fp32", float(fp[list(ch)].mean()), source=src)
    put("cpu.all123.int8", float(np.mean(list(i8.values()))), source=src)
    put_interval("cpu.all123", video_bootstrap(list(ch.values()), seed=0), src)
    put_interval("cpu.all121_excl_flips",
                 video_bootstrap([v for k, v in ch.items() if k not in ("person19_2", "uav4")], seed=0), src)
    put("cpu.all123.within_0.02", sum(abs(v) <= 0.02 for v in ch.values()), source=src)
    for v in ["uav2", "car4_s", "group1_2"]:
        put(f"cpu.loss.{v}", ch[v], source=src)
    g, a = p("fp32_gpu"), p("fp32_gpu_pgd4s10")
    put("cpu.pgd.clean_gpu", float(g[s40].mean()), source=src)
    put("cpu.pgd.attacked_gpu", float(a[s40].mean()), source=src)
    put_interval("cpu.pgd", video_bootstrap((a - g)[s40].values, seed=0), src)

    # recorded: frame-level analysis outputs and matched timing
    n5 = json.loads((ROOT / "tracks/cpu_int8/analysis/2026-10-22_round5-final/numbers5.json").read_text())
    s5 = "tracks/cpu_int8/analysis/2026-10-22_round5-final/numbers5.json"
    put("cpu.warn.auroc", n5["F"]["test_auroc"], "recorded", s5)
    put("cpu.warn.auroc.lo", n5["F"]["ci"][0], "recorded", s5)
    put("cpu.warn.auroc.hi", n5["F"]["ci"][1], "recorded", s5)
    put("cpu.warn.onsets", n5["F"]["onsets"], "recorded", s5)
    put("cpu.warn.alarm_before_onset", n5["F"]["alarm_before_onset"], "recorded", s5)
    put("cpu.attack.alarm_share_pct", 100 * n5["A"]["flagged"], "recorded", s5)
    for t, folder in [("4", "retime_matched_t4"), ("1", "retime_matched_t1")]:
        sj = f"tracks/cpu_int8/analysis/2026-11-02_retime-matched/{folder}/summary.json"
        s = json.loads((ROOT / sj).read_text())
        b = s["by_threads"][t]
        for k in ["fp32_net_ms", "int8bb-s1_net_ms", "fp32_step_ms", "int8bb-s1_step_ms", "fp32_backbone_ms",
                  "int8bb-s1_backbone_ms", "fp32_head_ms", "int8bb-s1_head_ms"]:
            if k in b:
                put(f"cpu.timing.t{t}.{k}", b[k]["median"], "recorded", sj)
        for k in ["speedup_net_ms", "speedup_step_ms"]:
            if k in b:
                put(f"cpu.timing.t{t}.{k}", b[k]["median"], "recorded", sj)
                put(f"cpu.timing.t{t}.{k}.min", b[k]["min"], "recorded", sj)
                put(f"cpu.timing.t{t}.{k}.max", b[k]["max"], "recorded", sj)
        if t == "4":
            put("cpu.size_mb.fp32", s["size_mb"]["fp32"], "recorded", sj)
            put("cpu.size_mb.int8", s["size_mb"]["int8bb-s1"], "recorded", sj)


# ---------------------------------------------------------------- domain
def domain():
    cl = {r["sequence"]: r["source_cluster"] for r in csv.DictReader(open(SPL / "domain/source_cluster_mapping.csv"))}
    for name, key in [("VisDrone34", "visdrone34"), ("UAV123_10fps_76", "uav123_10fps_76")]:
        src = f"results/domain/per_video_{name}.csv"
        d = pd.read_csv(RES / f"domain/per_video_{name}.csv")
        b0 = d[d.model == "B0"].set_index("video").success_auc
        r1 = d[(d.model == "REF") & (d.seed == 20261023)].set_index("video").success_auc
        r2 = d[(d.model == "REF") & (d.seed == 20261102)].set_index("video").success_auc
        vids = sorted(b0.index)
        put(f"domain.{key}.n", len(vids), source=src)
        put(f"domain.{key}.clusters", len({cl[v] for v in vids}), source=src)
        put(f"domain.{key}.B0", float(b0[vids].mean()), source=src)
        put(f"domain.{key}.REF_s1", float(r1[vids].mean()), source=src)
        put(f"domain.{key}.REF_s2", float(r2[vids].mean()), source=src)
        diff = ((r1[vids] + r2[vids]) / 2 - b0[vids]).values
        put_interval(f"domain.{key}.REF-B0", cluster_bootstrap(diff, [cl[v] for v in vids], seed=20261105), src)
        if key == "uav123_10fps_76":
            put_interval(f"domain.{key}.s1-s2",
                         cluster_bootstrap((r1[vids] - r2[vids]).values, [cl[v] for v in vids], seed=20261105), src)


def fmt(v, dp):
    return f"{v:.{dp}f}" if isinstance(v, float) or dp > 0 else str(int(round(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write results/numbers.json")
    args = ap.parse_args()
    night(); cpu(); domain()
    if args.write:
        (RES / "numbers.json").write_text(json.dumps(OUT, indent=1, sort_keys=True))
        print("wrote results/numbers.json")
    exp = json.loads((ROOT / "paper/expected_numbers.json").read_text())
    bad = 0
    for e in exp:
        if e["key"] not in OUT:
            print(f"MISSING  {e['key']}"); bad += 1; continue
        got = fmt(OUT[e["key"]]["value"] * e.get("scale", 1), e["dp"])
        want = e["paper"]
        ok = float(got) == float(want) if e.get("signed_zero_ok") else got.replace("-0.000", "0.000") == want.replace("-0.000", "0.000")
        ok = ok or abs(float(got) - float(want)) <= e.get("tol", 0) + 1e-12
        print(f"{'OK  ' if ok else 'FAIL'}  {e['key']:<52} paper {want:>9}  got {got:>9}  [{OUT[e['key']]['kind']}]  {e['where']}")
        bad += not ok
    print(f"\n{len(exp) - bad}/{len(exp)} paper numbers reproduced")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
