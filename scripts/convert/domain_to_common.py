"""Convert the domain track's sealed evaluation JSONs to the common long format.

Source: tracks/domain_ft/evidence/stage7/artifacts/b2-eval/STAGE2-SEALED-*.evaluation.json
Output: results/domain/per_video_VisDrone34.csv, results/domain/per_video_UAV123_10fps_76.csv
VisDrone keeps only the 34 sequences flagged in source_cluster_mapping.csv (one sequence
shares a source flight with REF training and is excluded).
"""
import csv, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
EV = ROOT / "tracks/domain_ft/evidence/stage7"
ARMS = [("B0", 0, "B0"), ("REF", 20261023, "TC2-P1-REF-001"), ("REF", 20261102, "TC2-P1-REF-S2-001")]
SETS = [("VisDrone2019-SOT-test-dev", "VISDRONE", "per_video_VisDrone34.csv"),
        ("UAV123@10fps", "UAV123", "per_video_UAV123_10fps_76.csv")]

def main():
    cmap = list(csv.DictReader(open(EV / "source_cluster_mapping.csv")))
    keep = {r["sequence"] for r in cmap if r["in_visdrone_disjoint_34"] == "1"}
    for bench, tag, name in SETS:
        out = ROOT / "results/domain" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["benchmark", "video", "frames", "model", "seed", "success_auc", "precision20", "toolkit"])
            for model, seed, arm in ARMS:
                d = json.load(open(EV / f"artifacts/b2-eval/STAGE2-SEALED-{tag}-{arm}-sigmoid.evaluation.json"))
                for seq, s in sorted(d["sequences"].items()):
                    if tag == "VISDRONE" and seq not in keep:
                        continue
                    w.writerow([bench, seq, s["frame_count"], model, seed,
                                repr(s["success_auc"]), repr(s["precision_at_20px"]), "own"])
        print("wrote", out)

if __name__ == "__main__":
    main()
