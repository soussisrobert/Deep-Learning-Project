"""Resolve an UNAPPROVED, BLOCKED Wave5 proposal. No training launch exists here."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hift_research.wave5_preparation import (W4_HEAD,CONFIG_SHA,digest,read_json,exclusive_json,verify_map,
    source_inventory,authentic_asset_inventory,initialization_identity,unapproved_proposal,assert_unapproved)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--experiment-id',required=True)
    parser.add_argument('--expires-utc',required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();expires=datetime.fromisoformat(args.expires_utc)
    if expires.tzinfo is None or expires<=datetime.now(timezone.utc):raise ValueError('future timezone-qualified expiry required')
    if not args.experiment_id or any(c not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.experiment_id):
        raise ValueError('safe unique experiment identifier required')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    try:
        subprocess.run(['git','merge-base','--is-ancestor',W4_HEAD,'HEAD'],cwd=ROOT,check=True)
        config=ROOT/'configs/B2-Pilot-W4.json'
        if digest(config)!=CONFIG_SHA:raise ValueError('do not rewrite frozen config expectation')
        cfg=read_json(config);binding=read_json(ROOT/'artifacts/wave4-lifecycle/final-config-binding-001.json')
        verify_map({ROOT/name:value for name,value in binding['critical_source_sha256'].items()})
        train=read_json(ROOT/cfg['data']['train_policy']);dev=read_json(ROOT/cfg['data']['development_policy'])
        references={'config':{'path':str(config),'sha256':CONFIG_SHA},
            'manifest':{'path':str(ROOT/cfg['data']['manifest']),'sha256':cfg['data']['manifest_sha256']},
            'train_policy':{'path':str(ROOT/cfg['data']['train_policy']),'sha256':cfg['data']['train_policy_sha256']},
            'development_policy':{'path':str(ROOT/cfg['data']['development_policy']),'sha256':cfg['data']['development_policy_sha256']},
            'development_cohort':{'path':str(ROOT/cfg['development']['cohort_manifest']),'sha256':cfg['development']['cohort_sha256']}}
        for split,policy in (('train',train),('development',dev)):
            verify_map(policy['source_code_hashes'])
            for key in ('population','reservations','source_content_screen','eligibility_prereg'):
                references[split+'_'+key]={'path':policy[key+'_path'],'sha256':policy[key+'_sha256']}
        for name,expected in [('backbone','933d95c2f25e0b9307ed4cf772ae6b338605cbbc430ea637e4f4ad5743b4c2e8'),
                              ('official_tracker','8c697d2b6d2d7aa36e8d035c89e6897b3b5733b94abe56509420329bc4ba3101')]:
            path=ROOT/('third_party/HiFT/pretrained_models/alexnet-bn.pth' if name=='backbone' else 'artifacts/checkpoints/general_model.pth')
            references[name]={'path':str(path),'sha256':expected}
        ledger=ROOT/'artifacts/wave4-preflight/PREFLIGHT-002/train/candidate_ledger.jsonl'
        references['train_preflight_ledger']={'path':str(ledger),'sha256':'bfa185342524cc8b2cf44f7551f92c93e77c1a8f9ff946d14cc6d32077ecfaa3'}
        verify_map({ref['path']:ref['sha256'] for ref in references.values()})
        initialization=initialization_identity(ROOT)
        inventory={'schema':'hift-wave5-preparation-inventory-v1','phase':'PREPARATION_ONLY_NO_REAL_SGD',
            'created_utc':datetime.now(timezone.utc).isoformat(),'wave4_handoff':W4_HEAD,
            'current_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'upstream':'992aa7c1605360d607af84cb804b93cd8ee81caf','source_sha256':source_inventory(ROOT),
            'wave4_critical_source_sha256':binding['critical_source_sha256'],
            'temporal_source_sha256':train['source_code_hashes'],'references':references,
            'authentic_assets':authentic_asset_inventory(ROOT),'initialization':initialization,
            'environment':{'python':sys.version,'interpreter':sys.executable,'interpreter_sha256':digest(sys.executable),
                'dependency_lock_sha256':digest(ROOT/'requirements/base.txt'),
                'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','opencv-python','pytest')}},
            'inventory_semantics':'Every named path is exact actual-byte SHA256; union all maps. No wildcard permissions. Additional source snapshot NOT claimed an enforced launch guard.'}
        inventory_path=output/'inventory.json';exclusive_json(inventory_path,inventory)
        proposal=unapproved_proposal(ROOT,args.experiment_id,{'path':str(inventory_path),'sha256':digest(inventory_path)},args.expires_utc,args.gpu_uuid)
        proposal.update({'training_manifest_sha256':cfg['data']['manifest_sha256'],'training_policy_sha256':cfg['data']['train_policy_sha256'],
            'development_policy_sha256':cfg['data']['development_policy_sha256'],'development_cohort_sha256':cfg['development']['cohort_sha256'],
            'reservations_sha256':cfg['data']['reservations_sha256'],'initial_parameters_sha256':initialization['all_parameters_sha256']})
        assert_unapproved(proposal);proposal_path=output/'UNAPPROVED-PROPOSAL.json';exclusive_json(proposal_path,proposal)
        receipt={'result':'PREPARATION_VERIFIED_EXECUTION_BLOCKED','proposal_path':str(proposal_path),'proposal_sha256':digest(proposal_path),
            'inventory_sha256':digest(inventory_path),'source_files':len(inventory['source_sha256']),'critical_files':len(binding['critical_source_sha256']),
            'authentic_frames':160,'accepted_archives':1151,'initial_parameters_sha256':initialization['all_parameters_sha256'],
            'cuda_initialized':False,'real_images_backward':0,'real_optimizer_steps':0,'training_attempts':0,'effective_updates':0,
            'approved_grant_created':False,'pilot_output_created':False,'elapsed_preparation_seconds':time.perf_counter()-start}
        exclusive_json(output/'receipt.json',receipt);print(json.dumps(receipt,indent=2))
    except Exception as exc:
        exclusive_json(output/'failure.json',{'type':type(exc).__name__,'error':str(exc),'no_training_launch':True,'elapsed_seconds':time.perf_counter()-start})
        raise

if __name__=='__main__':main()
