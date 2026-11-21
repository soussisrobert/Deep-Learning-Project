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


# ---------------------------------------------------------------- crop zoom
def crop_zoom():
    src = "results/crop_zoom/per_video_crop_zoom.csv"
    d = pd.read_csv(RES / "crop_zoom/per_video_crop_zoom.csv")
    for bench, key in [("UAV123@10fps(subsampled)", "uav123_10fps_sub"), ("UAV20L", "uav20l")]:
        p = d[d.benchmark == bench].pivot(index="video", columns="model", values="success_auc")
        put(f"zoom.{key}.n", len(p), source=src)
        for m in p.columns:
            put(f"zoom.{key}.{m}", float(p[m].mean()), source=src)
            if m != "base1.2":
                put_interval(f"zoom.{key}.{m}-base", video_bootstrap((p[m] - p["base1.2"]).values, seed=20261121), src)


# ---------------------------------------------------------------- box refinement (corner head)
def box_refinement():
    src = "results/box_refinement/per_video_VisDrone34.csv"
    cl = {r["sequence"]: r["source_cluster"] for r in csv.DictReader(open(SPL / "domain/source_cluster_mapping.csv"))}
    d = pd.read_csv(RES / "box_refinement/per_video_VisDrone34.csv")
    p = d.pivot(index="video", columns="model", values="success_auc")
    put("refine.visdrone34.n", len(p), source=src)
    put("refine.visdrone34.clusters", len({cl[v] for v in p.index}), source=src)
    for m in p.columns:
        put(f"refine.visdrone34.{m}", float(p[m].mean()), source=src)
    put_interval("refine.visdrone34.pipeline-stock",
                 cluster_bootstrap((p["HiFT_corner_pipeline"] - p["HiFT_stock_raw"]).values,
                                   [cl[v] for v in p.index], seed=20261105), src)


# ---------------------------------------------------------------- additional paper numbers (final draft)
def extras():
    T = ROOT / "tracks"
    # night: speed summary (job 55514), overlap screen header, containment share
    sp = (T / "night/evidence/E26_speed_summary.txt").read_text()
    for m in re.finditer(r"SPEED (\w+)\s+pass (\d): 20 videos, (\d+) frames, ([\d.]+) fps", sp):
        put(f"night.speed.{m.group(1)}_p{m.group(2)}.fps", float(m.group(4)), "recorded", "tracks/night/evidence/E26_speed_summary.txt")
        put(f"night.speed.{m.group(1)}_p{m.group(2)}.frames", int(m.group(3)), "recorded", "tracks/night/evidence/E26_speed_summary.txt")
    lk = (T / "night/evidence/sources/output_HiFT_uavdark_leak_49103.out").read_text()
    src = "tracks/night/evidence/sources/output_HiFT_uavdark_leak_49103.out"
    m = re.search(r"UAVDark135: (\d+) sequence folders, (\d+) frames", lk)
    put("night.leak.sequences", int(m.group(1)), "recorded", src)
    put("night.leak.total_frames", int(m.group(2)), "recorded", src)
    put("night.leak.max_nonmatch_corr", float(re.search(r"highest lowres corr among non-matches: ([\d.]+)", lk).group(1)), "recorded", src)
    put("night.E15.start_in_window_pct", 100 * OUT["night.E15.start_in_window"]["value"] / OUT["night.E15.stretches"]["value"],
        source="tracks/night/evidence/sources/E15_stretches.csv")

    # CPU: GPU vs CPU fp32 on TEST-40, whole-backbone draws, DEV selection, calibration crops, timing details
    d = pd.read_csv(RES / "cpu/per_video_UAV123.csv")
    s40 = sorted(json.loads((SPL / "cpu/uav123_test40.json").read_text())["sequences"])
    c = d[d.model == "fp32_cpu"].set_index("video").success_auc[s40]
    g = d[d.model == "fp32_gpu"].set_index("video").success_auc[s40]
    put_interval("cpu.cpu_minus_gpu_test40", video_bootstrap((c - g).values, seed=0), "results/cpu/per_video_UAV123.csv")
    put("cpu.cpu_minus_gpu_test40.person1_s", float(c["person1_s"] - g["person1_s"]), source="results/cpu/per_video_UAV123.csv")
    w = pd.read_csv(T / "cpu_int8/analysis/all-rounds/per_video_success.csv").set_index("video")
    srcw = "tracks/cpu_int8/analysis/all-rounds/per_video_success.csv"
    fpc = w.loc[s40, "uav123_fp32_clean_nodef_t4"]
    cols = ["uav123-sub40_int8bb_clean_nodef_t4"] + [f"uav123-sub40_int8bb_clean_nodef_t4_cal{k}" for k in (1, 2, 3)]
    put("cpu.wholebackbone_test40.mean4", float(np.mean([w.loc[s40, col].mean() - fpc.mean() for col in cols])), source=srcw)
    dev = sorted(json.loads((SPL / "cpu/uav123_dev20.json").read_text())["sequences"])
    fpd = w.loc[dev, "uav123-dev20_fp32_clean_nodef_t4"]
    excl = []
    for cand in ["int8bb_c20x2", "int8bb_c40x4s", "int8bb-s1_c20x2", "int8bb-s1_c40x4s"]:
        D = np.stack([w.loc[dev, f"uav123-dev20_{cand}_cal{k}"].values for k in (1, 2, 3)])
        ch = pd.Series(D.mean(0) - fpd.values, index=dev)
        put(f"cpu.dev20.{cand}.mean", float(ch.mean()), source=srcw)
        excl.append(float(ch.drop("person19_2").mean()))
        if cand == "int8bb-s1_c40x4s":
            put("cpu.dev20.selected.person19_2", float(ch["person19_2"]), source=srcw)
    put("cpu.dev20.excl_person19_2.min", min(excl), source=srcw)
    put("cpu.dev20.excl_person19_2.max", max(excl), source=srcw)
    cal = T / "cpu_int8/analysis/2026-10-22_round5-final/calibration_summaries"
    for name, key in [("uav123-sub40_int8sel_clean_nodef_t4_cal1", "test_cal1"), ("uav123-sub40_int8sel_clean_nodef_t4_cal2", "test_cal2"),
                      ("uav123-sub40_int8sel_clean_nodef_t4_cal3", "test_cal3"), ("uav123-nt83_int8sel_clean_nodef_t4_cal0", "nontest")]:
        js = json.loads((cal / f"{name}.summary.json").read_text())
        put(f"cpu.calib_crops.{key}", js["variant_info"]["calibration_crops"], "recorded", f"{cal.relative_to(ROOT)}/{name}.summary.json")
    ses = list(csv.DictReader(open(T / "cpu_int8/analysis/2026-11-02_retime-matched/retime_matched_t4/sessions.csv")))
    put("cpu.timing.t4.tracker_frames_per_session", int(ses[0][list(ses[0])[-1]]), "recorded",
        "tracks/cpu_int8/analysis/2026-11-02_retime-matched/retime_matched_t4/sessions.csv")
    put("cpu.timing.t4.head_share_pct", 100 * OUT["cpu.timing.t4.int8bb-s1_head_ms"]["value"] / OUT["cpu.timing.t4.int8bb-s1_net_ms"]["value"],
        "recorded", "tracks/cpu_int8/analysis/2026-11-02_retime-matched/retime_matched_t4/summary.json")

    # domain: validation raw vs sigmoid, grid size, training arms, REF schedule
    st = (T / "domain_ft/reports/TC2_STATE.md").read_text()
    m = re.search(r"B0-raw ([\d.]+)/.*?B0-sigmoid ([\d.]+)/", st)
    put("domain.val.B0_raw_M", float(m.group(1)), "recorded", "tracks/domain_ft/reports/TC2_STATE.md")
    put("domain.val.B0_sigmoid_M", float(m.group(2)), "recorded", "tracks/domain_ft/reports/TC2_STATE.md")
    put("domain.stage3_grid.settings", len(json.loads((T / "domain_ft/evidence/stage7/configs/stage3_track_grid.json").read_text())),
        "recorded", "tracks/domain_ft/evidence/stage7/configs/stage3_track_grid.json")
    co = (T / "domain_ft/reports/STAGE7_CLOSEOUT.md").read_text()
    arms = re.search(r"eight training arms \(([^)]*)\)", co).group(1).split(",")
    put("domain.training_arms_rejected", len(arms), "recorded", "tracks/domain_ft/reports/STAGE7_CLOSEOUT.md")
    for seed_dir in ["TC2-P1-REF-001", "TC2-P1-REF-S2-001"]:
        mf = json.loads((T / f"domain_ft/evidence/stage7/artifacts/tc2-runs/{seed_dir}/manifest.json").read_text())
        sc = mf["schedule"]
        src = f"tracks/domain_ft/evidence/stage7/artifacts/tc2-runs/{seed_dir}/manifest.json"
        put(f"domain.ref.{seed_dir}.updates", sc["epochs"] * sc["steps_per_epoch"], "recorded", src)
        put(f"domain.ref.{seed_dir}.batch", sc["batch"], "recorded", src)
        put(f"domain.ref.{seed_dir}.lr_first", sc["lr_table_used"][0], "recorded", src)
        put(f"domain.ref.{seed_dir}.lr_last", sc["lr_table_used"][-1], "recorded", src)

    # box refinement: clip counts, inventory, baseline check, validation and corner-stage increment
    cl = {r["sequence"]: r["source_cluster"] for r in csv.DictReader(open(SPL / "domain/source_cluster_mapping.csv"))}
    rf = pd.read_csv(RES / "box_refinement/per_video_VisDrone34.csv").pivot(index="video", columns="model", values="success_auc")
    dlt = rf["HiFT_corner_pipeline"] - rf["HiFT_stock_raw"]
    put("refine.visdrone34.better", int((dlt > 0.005).sum()), source="results/box_refinement/per_video_VisDrone34.csv")
    put("refine.visdrone34.worse", int((dlt < -0.005).sum()), source="results/box_refinement/per_video_VisDrone34.csv")
    b0 = pd.read_csv(RES / "domain/per_video_VisDrone34.csv")
    b0 = b0[b0.model == "B0"].set_index("video").success_auc[rf.index]
    put_interval("refine.baseline_check.stock_raw-B0_sigmoid",
                 cluster_bootstrap((rf["HiFT_stock_raw"] - b0).values, [cl[v] for v in rf.index], seed=20261105),
                 "results/box_refinement + results/domain per_video_VisDrone34.csv")
    H = T / "box_refinement/handoff"
    inv = pd.read_csv(H / "dataset_inventory.csv")
    for sp_, key in [("train", "train"), ("val", "val"), ("test", "test")]:
        sub = inv[inv.split == sp_]
        put(f"refine.inventory.{key}.clips", len(sub), "recorded", "tracks/box_refinement/handoff/dataset_inventory.csv")
        put(f"refine.inventory.{key}.sources", sub.source_cluster.nunique(), "recorded", "tracks/box_refinement/handoff/dataset_inventory.csv")
    S = {k: set(g.source_cluster) for k, g in inv.groupby("split")}
    put("refine.inventory.shared_sources", len(S["train"] & S["val"]) + len(S["train"] & S["test"]) + len(S["val"] & S["test"]),
        "recorded", "tracks/box_refinement/handoff/dataset_inventory.csv")
    v = json.loads((H / "evidence/corner_validation.json").read_text())["averages"]
    put("refine.val.pipeline", v["gentle"]["success_auc"], "recorded", "tracks/box_refinement/handoff/evidence/corner_validation.json")
    put("refine.val.coarse", v["version_one"]["success_auc"], "recorded", "tracks/box_refinement/handoff/evidence/corner_validation.json")
    hs = json.loads((H / "evidence/historical_corner_test35.json").read_text())["sequences"]
    keep = sorted(rf.index)
    put("refine.test34.corner_over_coarse_historical",
        float(np.mean([hs[x]["gentle"]["success_auc"] - hs[x]["version_one"]["success_auc"] for x in keep])),
        "recorded", "tracks/box_refinement/handoff/evidence/historical_corner_test35.json (34 retained clips)")
    rp = json.loads((H / "evidence/replay_per_video.json").read_text())
    put("refine.test34.corner_over_coarse_replay",
        float(np.mean([rp[x]["scores"]["corner_pipeline"]["success_auc"] - rp[x]["scores"]["coarse_refiner"]["success_auc"] for x in keep])),
        "recorded", "tracks/box_refinement/handoff/evidence/replay_per_video.json")


def fmt(v, dp):
    return f"{v:.{dp}f}" if isinstance(v, float) or dp > 0 else str(int(round(v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write results/numbers.json")
    args = ap.parse_args()
    night(); cpu(); domain(); crop_zoom(); box_refinement(); extras()
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
