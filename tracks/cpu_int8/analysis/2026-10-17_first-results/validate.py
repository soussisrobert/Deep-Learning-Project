import json, sys, glob, os
import numpy as np
B = sys.argv[1]
sub = json.load(open(f"{B}/attack-subset-40.json"))
S40 = sub["sequences"]
runs = {}
for f in sorted(glob.glob(f"{B}/results/*/summary.json")):
    d = os.path.dirname(f); n = os.path.basename(d)
    runs[n] = (json.load(open(f)), json.load(open(f"{d}/per_seq.json")), d)
print("runs:", list(runs))
issues = []
for n, (s, ps, d) in runs.items():
    names = [p["name"] for p in ps]
    succ = np.mean([p["success_curve"] for p in ps], axis=0); prec = np.mean([p["precision_curve"] for p in ps], axis=0)
    ok_auc = abs(succ.mean() - s["success_auc"]) < 1e-4 and abs(prec[20] - s["precision_20"]) < 1e-4
    npred = len(glob.glob(f"{d}/preds/*.txt"))
    lines = sum(sum(1 for _ in open(f"{d}/preds/{x}.txt")) for x in names)
    fr = sum(p["frames"] for p in ps)
    is40 = "sub40" in n
    setok = (names == S40) if is40 else (len(names) == 123)
    print(f"{n:42s} seqs {len(names):3d} frames {fr:6d} predlines {lines:6d} preds {npred:3d} agg_ok {ok_auc} set_ok {setok} "
          f"auc {s['success_auc']:.4f} p20 {s['precision_20']:.4f} model={s.get('model',{}).get('snapshot')} dev={s['device']} "
          f"ms/frame={ {k: round(v,2) for k,v in s['ms_per_frame'].items()} } minutes={s['total_minutes']:.1f}")
    if not (ok_auc and setok and fr == lines and npred == len(names)): issues.append(n)
    if is40 and sum(p["frames"] for p in ps) != sub["frame_count"]: issues.append(n + " framecount")
print("issues:", issues)
# frame sets identical across sub40
fr40 = {n: {p["name"]: p["frames"] for p in ps} for n, (s, ps, d) in runs.items() if "sub40" in n}
base = fr40["uav123-sub40_fp32_clean_nodef"]
print("sub40 frame maps identical:", all(v == base for v in fr40.values()))
# GPU vs CPU full clean
g = {p["name"]: p for p in runs["uav123_fp32_clean_nodef"][1]}; c = {p["name"]: p for p in runs["uav123_fp32_clean_nodef_t4"][1]}
dd = np.array([c[k]["success_auc"] - g[k]["success_auc"] for k in g])
print(f"CPU-GPU full clean per-seq AUC diff: mean {dd.mean():+.4f} |max| {np.abs(dd).max():.4f} seqs differing>0.01: {(np.abs(dd)>0.01).sum()} identical(<1e-6): {(np.abs(dd)<1e-6).sum()}")
big = sorted(((k, round(c[k]['success_auc']-g[k]['success_auc'],3)) for k in g), key=lambda x: -abs(x[1]))[:5]; print("largest", big)
# sub40 clean vs full GPU clean on same 40
s40 = {p["name"]: p for p in runs["uav123-sub40_fp32_clean_nodef"][1]}
d2 = np.array([s40[k]["success_auc"] - g[k]["success_auc"] for k in S40])
print(f"sub40 clean vs full-run clean on same 40: mean {d2.mean():+.4f} |max| {np.abs(d2).max():.4f} n>0.01 {(np.abs(d2)>0.01).sum()}")
print("full-GPU clean restricted to 40:", np.mean([np.mean(g[k]['success_curve']) for k in S40]))
# per-seq frame counts in full vs 40
print("frames in subset == full-run frames:", all(g[k]["frames"] == s40[k]["frames"] for k in S40))
