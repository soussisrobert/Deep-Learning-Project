"""Final preservation/receipt audit; no model, tracking, downloads or SGD."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hift_research.evaluation.assets import read_json,sha256_file,write_json_exclusive
from hift_research.provenance import source_identity
from hift_research import OFFICIAL_CHECKPOINT_SHA256,UPSTREAM_COMMIT


def main():
    receipt=read_json(ROOT/'experiments/wave3_validation.json')
    verified={}
    for name,ref in receipt['references'].items():
        if sha256_file(ROOT/ref['path'])!=ref['sha256']:raise ValueError('changed Wave3 receipt reference:'+name)
        verified[name]=True
    old=read_json(ROOT/'experiments/wave2_validation.json')
    for name,ref in old['references'].items():
        if sha256_file(ref['path'])!=ref['sha256']:raise ValueError('changed Wave2 artifact:'+name)
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    if git('-C','third_party/HiFT','rev-parse','HEAD')!=UPSTREAM_COMMIT:raise ValueError('upstream changed')
    if git('-C','third_party/HiFT','status','--porcelain'):raise ValueError('upstream dirty')
    baseline_files=('configs/B0.json','configs/B1-score.json','configs/B2.json','configs/R1.json',
        'scripts/run_sequence.py','src/hift_research/inference.py','src/hift_research/geometry.py',
        'src/hift_research/diagnostics/policy.py','src/hift_research/training/graph.py',
        'src/hift_research/training/profiling.py','experiments/wave2_validation.json',
        'reports/wave2_progress.md','reports/wave2_m1.md','reports/wave2_training.md')
    base='fba88e5b3452bf6e92980f14a52604d5d93eb0b7'
    for path in baseline_files:
        original=subprocess.check_output(['git','show',f'{base}:{path}'],cwd=ROOT)
        if original!=(ROOT/path).read_bytes():raise ValueError('immutable baseline file changed:'+path)
    if sha256_file(ROOT/'artifacts/checkpoints/general_model.pth')!=OFFICIAL_CHECKPOINT_SHA256:raise ValueError('checkpoint changed')
    cfg=read_json(ROOT/'configs/B2-wave3-proposed.json')
    if any(cfg[key] for key in ('launch_allowed','real_data_sgd_authorized','validated_real_data')):raise ValueError('training inadvertently enabled')
    for path,digest in cfg['manifest_source_code_hashes'].items():
        if sha256_file(ROOT/path)!=digest:raise ValueError('frozen input code changed:'+path)
    output={'kind':'Wave3_final_preservation_and_receipt_verification_no_experiments',
        'created_utc':datetime.now(timezone.utc).isoformat(),'source':source_identity(ROOT),
        'receipt_reference_count':len(verified),'receipt_references_verified':verified,
        'old_reference_count':len(old['references']),'baseline_files_bitexact_to_start':list(baseline_files),
        'upstream_clean':True,'checkpoint_sha256':OFFICIAL_CHECKPOINT_SHA256,
        'training_launch_allowed':False,
        'observed_integrated_test_result':{'command':'.venv/Scripts/python.exe -m pytest -q',
             'passed':362,'runtime_seconds':24.58,'host_job':'bash-60d408b069ff495e',
             'provenance':'observed completed tool result; no claim this audit reruns tests'},
        'dependency_check':'No broken requirements found.',
        'no_model_execution_download_or_optimization':True}
    write_json_exclusive(ROOT/'artifacts/wave3-diagnostics/FINAL-VERIFICATION-001.json',output)
    print(f'PASS:{len(verified)} Wave3 refs,{len(old["references"])} Wave2 refs,{len(baseline_files)} immutable baseline files; upstream/checkpoint/config bindings intact')


if __name__=='__main__':main()
