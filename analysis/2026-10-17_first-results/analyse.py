import json, sys, glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
B, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
S40 = json.load(open(f"{B}/attack-subset-40.json"))["sequences"]
def load(n):
    d=f"{B}/results/{n}"; return json.load(open(f"{d}/summary.json")), {p["name"]:p for p in json.load(open(f"{d}/per_seq.json"))}
P="uav123-sub40_fp32_"
conds=[("clean","clean_nodef",0),("noise",'noise2_nodef',2),("noise","noise4_nodef",4),("noise","noise8_nodef",8),
       ("fgsm","fgsm1_nodef",1),("fgsm","fgsm2_nodef",2),("fgsm","fgsm4_nodef",4),("fgsm","fgsm8_nodef",8),("pgd","pgd4s10_nodef",4)]
R={}
for a,suf,e in conds: R[(a,e)]=load(P+suf)
clean=R[("clean",0)][1]
rng=np.random.default_rng(0)
def boot(d,n=10000):
    idx=rng.integers(0,len(d),(n,len(d))); m=d[idx].mean(1); return np.percentile(m,[2.5,97.5])
rows=[]
for (a,e),(s,ps) in R.items():
    d_clean=np.array([ps[k]["success_auc"]-clean[k]["success_auc"] for k in S40])
    row=dict(cond=f"{a} eps{e}" if a!="clean" else "clean", attack=a, eps=e, auc=s["success_auc"], p20=s["precision_20"],
             d_clean=d_clean.mean(), ci_clean=boot(d_clean) if a!="clean" else (0,0),
             worse05=int((d_clean<-0.05).sum()), better05=int((d_clean>0.05).sum()), med=np.median(d_clean),
             ms_total=s["ms_per_frame"]["total"], ms_attack=s["ms_per_frame"]["attack"], ms_model=s["ms_per_frame"]["model"],
             minutes=s["total_minutes"], obj=s.get("attack_objective"))
    if (a in ("fgsm","pgd")) and ("noise",e) in R:
        nz=R[("noise",e)][1]; dn=np.array([ps[k]["success_auc"]-nz[k]["success_auc"] for k in S40])
        row.update(d_noise=dn.mean(), ci_noise=boot(dn), worse_than_noise05=int((dn<-0.05).sum()))
    rows.append(row)
for r in rows:
    print(f"{r['cond']:10s} AUC {r['auc']:.3f} P20 {r['p20']:.3f} dAUCvsClean {r['d_clean']:+.3f} CI[{r['ci_clean'][0]:+.3f},{r['ci_clean'][1]:+.3f}] median {r['med']:+.3f} "
          f"worse>0.05 {r['worse05']}/40 better>0.05 {r['better05']}/40 "
          + (f"vsNoise {r['d_noise']:+.3f} CI[{r['ci_noise'][0]:+.3f},{r['ci_noise'][1]:+.3f}] worse-than-noise {r['worse_than_noise05']}/40 " if 'd_noise' in r else "")
          + f"| ms/frame total {r['ms_total']:.1f} attack {r['ms_attack']:.1f} model {r['ms_model']:.1f} | {r['minutes']:.0f} min | obj {r['obj']}")
json.dump([{k:(list(map(float,v)) if isinstance(v,(tuple,np.ndarray)) else (float(v) if isinstance(v,(np.floating,)) else v)) for k,v in r.items()} for r in rows], open(f"{OUT}/attack_table.json","w"), indent=1, default=float)
# per-category PGD drop
cat=lambda k: re.match(r"[a-z]+", k).group(0)
pgd=R[("pgd",4)][1]; nz4=R[("noise",4)][1]
cats={}
for k in S40: cats.setdefault(cat(k),[]).append(k)
print("\nPGD eps4 per category (n, clean, noise4, pgd):")
for c,ks in sorted(cats.items()):
    print(f"  {c:10s} n={len(ks):2d} clean {np.mean([clean[k]['success_auc'] for k in ks]):.3f} noise4 {np.mean([nz4[k]['success_auc'] for k in ks]):.3f} pgd {np.mean([pgd[k]['success_auc'] for k in ks]):.3f}")
# where clean is already failing
lowc=[k for k in S40 if clean[k]["success_auc"]<0.2]
print(f"\nseqs with clean AUC<0.2: {len(lowc)}; PGD on the other {40-len(lowc)}: clean {np.mean([clean[k]['success_auc'] for k in S40 if k not in lowc]):.3f} -> pgd {np.mean([pgd[k]['success_auc'] for k in S40 if k not in lowc]):.3f}")
print("PGD: seqs keeping >=50% of clean AUC:", sum(pgd[k]['success_auc']>=0.5*clean[k]['success_auc'] for k in S40 if clean[k]['success_auc']>=0.2), "of", 40-len(lowc))
# precision rise with fgsm1
f1=R[("fgsm",1)][1]; dp=np.array([f1[k]["precision_20"]-clean[k]["precision_20"] for k in S40]); da=np.array([f1[k]["success_auc"]-clean[k]["success_auc"] for k in S40])
print(f"\nFGSM eps1: dP20 mean {dp.mean():+.3f} (up in {(dp>0.01).sum()}, down in {(dp<-0.01).sum()}); dAUC mean {da.mean():+.3f}; seqs with P20 up & AUC down: {((dp>0.01)&(da<-0.01)).sum()}")
# ---------- figures ----------
C={"noise":"#52514e","fgsm":"#2a78d6","pgd":"#eb6834","clean":"#0b0b0b"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#a3a29d","axes.grid":True,"grid.color":"#e7e6e2","grid.linewidth":0.6})
fig,ax=plt.subplots(figsize=(6.2,3.8))
base=R[("clean",0)][0]["success_auc"]
ax.axhline(base,color=C["clean"],lw=1,ls="--"); ax.text(0.05,base+0.012,"clean (no attack) 0.582",va="bottom",fontsize=9,color="#52514e")
for a in ("noise","fgsm"):
    pts=sorted((e,R[(a,e)][0]["success_auc"]) for (aa,e) in R if aa==a)
    ax.plot(*zip(*pts),marker="o",ms=7,lw=2,color=C[a],label=("random noise" if a=="noise" else "FGSM (1 step)"),mec="white",mew=1.5)
    ax.annotate(("random noise" if a=="noise" else "FGSM"),pts[-1],xytext=(6,0),textcoords="offset points",va="center",color="#0b0b0b",fontsize=9)
ax.plot([4],[R[("pgd",4)][0]["success_auc"]],marker="D",ms=8,color=C["pgd"],mec="white",mew=1.5,ls="none",label="PGD (10 steps)")
ax.annotate("PGD-10  0.095",(4,R[("pgd",4)][0]["success_auc"]),xytext=(8,0),textcoords="offset points",va="center",fontsize=9)
ax.set(xlabel="perturbation budget ε (pixel levels out of 255)",ylabel="success AUC (0–1)",ylim=(0,0.65),xticks=[0,1,2,4,8],xlim=(-0.3,9.6),
       title="HiFT on 40 UAV123 sequences (A40 GPU, one seed)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_robustness.png",dpi=200); plt.close(fig)
# fig2: paired per-sequence, sorted by clean
order=sorted(S40,key=lambda k: clean[k]["success_auc"])
fig,ax=plt.subplots(figsize=(7.2,3.9)); x=np.arange(40)
for k_i,k in enumerate(order):
    ax.plot([k_i,k_i],[clean[k]["success_auc"],pgd[k]["success_auc"]],color="#c3c2b7",lw=1,zorder=1)
ax.scatter(x,[nz4[k]["success_auc"] for k in order],s=22,color=C["noise"],label="noise ε=4",zorder=2,edgecolor="white",lw=0.8)
ax.scatter(x,[clean[k]["success_auc"] for k in order],s=26,color=C["clean"],label="clean",zorder=3,edgecolor="white",lw=0.8)
ax.scatter(x,[pgd[k]["success_auc"] for k in order],s=30,marker="D",color=C["pgd"],label="PGD-10 ε=4",zorder=4,edgecolor="white",lw=0.8)
ax.set(xticks=x,ylabel="success AUC per sequence",ylim=(0,1),title="Same 40 sequences, paired: PGD vs noise at the same budget")
ax.set_xticklabels(order,rotation=90,fontsize=7); ax.legend(frameon=False,fontsize=9,loc="upper left",ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_paired_sequences.png",dpi=200); plt.close(fig)
# fig3: edge cost
E={}
for f in glob.glob(f"{B}/results/edge/*.json"):
    j=json.load(open(f)); E[(j["variant"],j["threads"])]=j
vs=["fp32","int8dyn","prune50","int8bb"]; lab={"fp32":"fp32","int8dyn":"int8 (transformer only)","prune50":"50% pruned","int8bb":"int8 backbone"}
cols={"fp32":"#52514e","int8dyn":"#1baf7a","prune50":"#6250d6","int8bb":"#2a78d6"}
fig,ax=plt.subplots(figsize=(6.2,3.6)); w=0.19
for i,v in enumerate(vs):
    vals=[E[(v,t)]["latency_ms_mean"] for t in (1,2,4)]
    b=ax.bar(np.arange(3)+(i-1.5)*w,vals,w-0.02,color=cols[v],label=lab[v])
    for xx,vv in zip(np.arange(3)+(i-1.5)*w,vals): ax.text(xx,vv+2,f"{vv:.0f}",ha="center",fontsize=7,color="#52514e")
ax.set(xticks=range(3),ylabel="network latency per frame (ms)",title="CPU cost per frame: Xeon Gold 6230R, 1/2/4 threads")
ax.set_xticklabels(["1 thread","2 threads","4 threads"]); ax.legend(frameon=False,fontsize=8,ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_cpu_latency.png",dpi=200); plt.close(fig)
for v in vs:
    j=E[(v,4)]; print(f"edge {v:8s} t4 lat {j['latency_ms_mean']:.1f} p90 {j['latency_ms_p90']:.1f} bb {j['backbone_ms_mean']:.1f} head {j['head_ms_mean']:.1f} size {j['size_mb']:.1f} nonzero {j['nonzero_params_m']:.2f}M/{j['params_m']:.2f}M peak {j['same_peak_fraction']:.3f} relerr {j['score_rel_error']:.3f} iters {j['iters']} t1 {E[(v,1)]['latency_ms_mean']:.1f} t2 {E[(v,2)]['latency_ms_mean']:.1f}")
print("cpu", E[("fp32",1)]["cpu"], "torch", E[("fp32",1)]["torch"], "weights", E[("fp32",1)]["weights"], "crops?", E[("fp32",1)].get("calibration_crops"), E[("int8bb",1)].get("calibration_crops"))
