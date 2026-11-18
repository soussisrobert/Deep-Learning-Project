"""Read-only Wave4 evidence/preservation/config authorization audit; no model/SGD."""
import argparse
import json
from pathlib import Path
import subprocess
from hift_research.evaluation.assets import read_json,sha256_file,write_json_exclusive
from hift_research.training.recipe import ROOT
from hift_research.training.wave4_recipes import validate_frozen_pilot,validate_temporal_binding
from hift_research.training.wave4_pilot import future_authorization
from hift_research.temporal_contract.population import normalize_source
from hift_research.temporal_contract.policy import load_policy
from hift_research.wave4_integrity import verify_prior


def verify():
    evidence=read_json(ROOT/'experiments/wave4_validation.json')
    checked={}
    for name,ref in evidence['references'].items():
        path=ROOT/ref['path']
        if sha256_file(path)!=ref['sha256']:raise ValueError('Wave4 evidence reference changed:'+name)
        checked[name]=ref['sha256']
    config_path=ROOT/evidence['references']['pilot_config']['path'];config=read_json(config_path)
    validate_frozen_pilot(config);dataset=validate_temporal_binding(config)
    if len(dataset)!=1600 or config['budget']['max_wall_seconds']!=900:raise ValueError('pilot budget binding incomplete')
    if not config['readiness']['recommended_GO'] or config['current_wave_real_sgd_authorized'] or config['future_authorization_file_created']:
        raise ValueError('recommendation/authorization boundaries invalid')
    binding=read_json(ROOT/evidence['references']['latest_final_config_binding']['path'])
    if sha256_file(config_path)!=binding['final_config_sha256']:raise ValueError('final config SHA drift')
    for path,expected in binding['critical_source_sha256'].items():
        if sha256_file(ROOT/path)!=expected:raise ValueError('critical tested code drift:'+path)
    manifest=read_json(ROOT/evidence['references']['source_manifest']['path'])
    normalized,quarantine=normalize_source(manifest)
    frame_count=sum(len(video['frames']) for video in normalized.values())
    if frame_count!=160:raise ValueError('actual source frame count changed')
    for name in ('train_policy','development_policy'):load_policy(ROOT/evidence['references'][name]['path'])
    def ref_json(name):return read_json(ROOT/evidence['references'][name]['path'])
    train,dev=ref_json('train_summary'),ref_json('development_summary')
    if (train['candidate_count'],train['accepted_positive_count'],train['accepted_negative_count'],train['rejected_count'],train['replacement_count'])!=(1024,973,50,1,0):
        raise ValueError('preflight class/rejection accounting contradiction')
    if (dev['candidate_count'],dev['accepted_positive_count'],dev['accepted_negative_count'],dev['replacement_count'])!=(128,64,64,0):
        raise ValueError('development cohort changed')
    for name in ('latest_noncontiguous_production_CUDA','shared_controller_synthetic_prior'):
        if not all(ref_json(name)['comparisons'].values()):raise ValueError('synthetic execution proof failed:'+name)
    if not all(ref_json('actual_loader')['comparisons'].values()):raise ValueError('actual loader proof failed')
    real=ref_json('actual_real_forward_only')
    if (real['result']!='passed_forward_only_no_SGD' or not real['parameters_unchanged'] or not real['no_parameter_gradients']
            or real['parameters_sha256_before']!=real['parameters_sha256_after'] or real['real_data_optimizer_steps'] or real['real_data_backward_calls']):
        raise ValueError('real no-SGD forward proof invalid')
    if real['development_initial_forward_only']['attempts']!=128:raise ValueError('complete intended DEV path not exercised')
    try:future_authorization(config_path,None,ack_real_sgd=True)
    except PermissionError as exc:denial=str(exc)
    else:raise ValueError('GO accidentally grants current real-SGD authority')
    if evidence['tests']['failures'] or evidence['tests']['errors']:raise ValueError('integrated tests failed')
    subprocess.run(['git','merge-base','--is-ancestor',evidence['tested_implementation_head'],'HEAD'],cwd=ROOT,check=True)
    preservation=verify_prior(ROOT)
    return {'result':'passed','current_HEAD':preservation['current_HEAD'],'current_git_status':preservation['current_git_status'],
        'tested_implementation_head':evidence['tested_implementation_head'],'protected_prior_file_count':preservation['protected_prior_file_count'],
        'actual_prior_working_bytes_unchanged':preservation['actual_working_bytes_equal_wave4_start'],
        'prior_reference_count':sum(len(value) for value in preservation['prior_receipt_references'].values()),
        'wave4_reference_count':len(checked),'wave4_reference_sha256':checked,'actual_source_frames_rehashed':frame_count,
        'critical_current_code_byteexact_to_layout_CUDA_proof':True,'critical_source_sha256':binding['critical_source_sha256'],
        'final_config_sha256':sha256_file(config_path),'integrated_tests':evidence['tests'],
        'recommendation':'GO_FOR_SEPARATELY_AUTHORIZED_BOUNDED_GRADER_ONLY_PILOT','real_SGD_authorized':False,
        'missing_future_grant_denial':denial,'real_pilot_started':False,'R1_developed':False,
        'no_prior_experiment_rerun_by_this_verifier':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();value=verify()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);write_json_exclusive(args.output,value)
    print(json.dumps({key:value[key] for key in ('result','current_HEAD','tested_implementation_head','protected_prior_file_count',
        'actual_prior_working_bytes_unchanged','prior_reference_count','wave4_reference_count','integrated_tests','final_config_sha256','real_SGD_authorized')},indent=2))
