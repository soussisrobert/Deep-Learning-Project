"""Freeze compact Wave4 evidence from existing immutable receipts; no experiment runs."""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from hift_research.evaluation.assets import read_json,sha256_file
from hift_research.training.recipe import ROOT

REFERENCES={
    'start_preservation':'artifacts/wave4-integration/START-001.json',
    'source_selection':'artifacts/wave4-source/SELECTION-001.json',
    'source_manifest':'artifacts/wave4-source/SOURCE-001/temporal_manifest.json',
    'source_independent_audit':'artifacts/wave4-source/SOURCE-001/independent_audit.json',
    'eligibility_preregistration':'artifacts/wave4-preflight/ELIGIBILITY-PREREG-001.json',
    'frozen_contract':'artifacts/wave4-preflight/FROZEN-002/freeze_receipt.json',
    'train_policy':'artifacts/wave4-preflight/FROZEN-002/train_policy.json',
    'development_policy':'artifacts/wave4-preflight/FROZEN-002/development_policy.json',
    'train_summary':'artifacts/wave4-preflight/PREFLIGHT-002/train/summary.json',
    'train_ledger':'artifacts/wave4-preflight/PREFLIGHT-002/train/candidate_ledger.jsonl',
    'development_summary':'artifacts/wave4-preflight/PREFLIGHT-002/development/summary.json',
    'development_ledger':'artifacts/wave4-preflight/PREFLIGHT-002/development/candidate_ledger.jsonl',
    'all_train_replay':'artifacts/wave4-preflight/VERIFICATION-002/train_verification.json',
    'all_development_replay':'artifacts/wave4-preflight/VERIFICATION-002/development_verification.json',
    'pair_distributions':'artifacts/wave4-preflight/VERIFICATION-002/distributions_supplement.json',
    'serialization_failure':'artifacts/wave4-preflight/PREFLIGHT-001/failure_receipt.json',
    'failed_source_snapshot':'artifacts/wave4-preflight/FAILED-001-SOURCE/snapshot_receipt.json',
    'serialization_value_parity':'artifacts/wave4-preflight/FORENSIC-001/comparison.json',
    'independent_GT_grid_algebra':'artifacts/wave4-integration/PAIR-ALGEBRA-001.json',
    'actual_loader':'artifacts/wave4-integration/REAL-LOADER-002/manifest.json',
    'actual_loader_failed001':'artifacts/wave4-integration/REAL-LOADER-001/failure.json',
    'device_epoch0_epoch10_synthetic':'artifacts/wave4-lifecycle/cuda-001/verification.json',
    'shared_controller_synthetic_prior':'artifacts/wave4-lifecycle/production-cuda-001/verification.json',
    'actual_forward_failed001':'artifacts/wave4-feasibility/REAL-FORWARD-001/failure.json',
    'latest_noncontiguous_production_CUDA':'artifacts/wave4-lifecycle/production-layout-cuda-001/verification.json',
    'latest_final_config_binding':'artifacts/wave4-lifecycle/final-config-binding-001.json',
    'actual_real_forward_only':'artifacts/wave4-feasibility/REAL-FORWARD-002/manifest.json',
    'budget':'artifacts/wave4-feasibility/BUDGET-001.json',
    'initial_plan':'experiments/wave4_feasibility_plan.json',
    'layout_dev_prereg_addendum':'experiments/wave4_layout_and_dev_addendum.json',
    'reference_config':'configs/B2-Reference-W4.json',
    'pilot_config':'configs/B2-Pilot-W4.json',
    'integrated_tests':'artifacts/wave4-integration/FINAL-TEST-001.xml',
}


def compile_evidence(implementation_head):
    refs={key:{'path':path,'sha256':sha256_file(ROOT/path)} for key,path in REFERENCES.items()}
    train=read_json(ROOT/REFERENCES['train_summary']);dev=read_json(ROOT/REFERENCES['development_summary'])
    real=read_json(ROOT/REFERENCES['actual_real_forward_only']);budget=read_json(ROOT/REFERENCES['budget'])
    binding=read_json(ROOT/REFERENCES['latest_final_config_binding'])
    config=read_json(ROOT/REFERENCES['pilot_config'])
    if sha256_file(ROOT/REFERENCES['pilot_config'])!=binding['final_config_sha256']:raise ValueError('final config binding drift')
    if config['budget']['max_wall_seconds']!=900 or not config['readiness']['recommended_GO']:raise ValueError('recipe not finalized')
    for path,expected in binding['critical_source_sha256'].items():
        if sha256_file(ROOT/path)!=expected:raise ValueError('tested critical source changed:'+path)
    tree=ET.parse(ROOT/REFERENCES['integrated_tests'])
    suites=list(tree.getroot().iter('testsuite'))
    tests={key:sum(int(suite.get(key,0)) for suite in suites) for key in ('tests','failures','errors','skipped')}
    tests['passed']=tests['tests']-tests['failures']-tests['errors']-tests['skipped']
    if tests['failures'] or tests['errors']:raise ValueError('integrated tests failed')
    return {'schema_version':1,'wave':4,'starting_delivery':'28e9e197c62a21450587465e4c32abdf97fb88d3',
        'tested_implementation_head':implementation_head,'ending_HEAD_policy':'Exact subsequent clean bookkeeping delivery HEAD is printed in final response and ignored DELIVERY-END receipt, not self-embedded in its own commit',
        'upstream_commit':'992aa7c1605360d607af84cb804b93cd8ee81caf',
        'recommendation':'GO_SEPARATELY_HUMAN_AUTHORIZED_BOUNDED_GRADER_ONLY_RELIABILITY_PILOT',
        'real_SGD_authorized':False,'real_optimizer_steps':0,'real_backward_calls':0,'real_pilot_started':False,
        'R1_developed':False,'frozen_B0_B1_rerun_or_retuned':False,
        'gates':{'W4-G1':'PASS_bounded_authentic_temporal','W4-G2':'PASS_bounded_simultaneous_known_distinct',
                 'W4-G3':'PASS_all_1024_plus128_audited','W4-G4':'PASS_serial_atomic_boundary_fallback','W4-G5':'PASS_recommendation_NOT_authorization'},
        'data':{'original_frames':160,'full_GT_rows':65877,'train_videos':['MOT17-02','MOT17-09','MOT17-10'],
            'development_videos':['MOT17-05'],'identity_scope':'author trajectory fragment within canonical release/video, NOT global person',
            'train_attempts':train['candidate_count'],'accepted_positive':train['accepted_positive_count'],
            'accepted_negative':train['accepted_negative_count'],'rejected':train['rejected_count'],'replacements':0,
            'nonzero_gap_positives':train['nonzero_gap_positives'],'safe_negative_actual_exclusion_proved':train['safe_negative_actual_exclusion_proved'],
            'development_attempts':128,'development_positive':dev['accepted_positive_count'],'development_negative':dev['accepted_negative_count'],
            'development_unique_negative_pairs':38,'development_negative_repeated_draws':26,
            'unknown_alias_policy':'No crossvideo/image different-ID negative; known metadata family reservations +SHA/pHash are not universal person/re-edit/pretraining proof'},
        'execution':{'synthetic_CUDA_total_effective_updates':16,'synthetic_sequence':[8,4,4],
            'latest_production_input_kind':'random-only noncontiguous HWC-to-CHW NumPy arrays canonicalized by same production NCHW bridge',
            'real_successful_train_forward_pairs':32,'real_complete_development_forward_pairs':128,'real_failed_prior_forward_attempts':4,
            'real_initial_dev_loss':real['development_initial_forward_only']['total_loss'],
            'real_parameters_unchanged':real['parameters_unchanged'],'real_no_parameter_gradients':real['no_parameter_gradients'],
            'no_adapted_checkpoint_saved':True,'safe_resume':'num_workers0/no-prefetch; atomic committed effective-window boundary only; tainted in-flight state discarded and entire indexed window replayed'},
        'pilot':{'max_updates':100,'max_attempt_slots':1600,'max_wall_seconds':900,'physical_attempt_slots':4,'effective_attempt_slots':16,
            'source_epoch':0,'grader_lr':.0025,'trainable_scope':'entire corrected grader fusion/transformer/prediction heads; frozen backbone and backboneBN',
            'future_explicit_human_grant_required':True,'no_future_grant_created':True,
            'central_projected_seconds':budget['central_seconds'],'engineering_range_seconds':budget['engineering_range_seconds'],
            'primary_checkpoint':'fixed last update100; bestDEVsecondary earliesttie cannotrescue',
            'signal':'fixedlast DEVloss<=.95initial, distinct from reliability/nondivergence; nottrackingaccuracy',
            'reference70epoch_fullrecipe_ready':False},
        'tests':tests,'references':refs,
        'limitations':['No real SGD or learned corrected-B2 checkpoint or convergence evidence',
            'No full70epoch/original4dataset/physical220batch or original2021runtime reproduction',
            'No exact asynchronous worker resume; manual recovery needs cumulative failedwork/time budget review',
            'DEV is a repeated weighted control, not128independent examples; one heldout known collection only',
            'Torch allocator is not total VRAM; short synthetic proof is not a long-run leak test',
            'Local ignored raw assets/arrays/models/run snapshots must be separately preserved and transferred']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation-head',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/wave4_validation.json')
    args=parser.parse_args()
    subprocess.run(['git','cat-file','-e',args.implementation_head+'^{commit}'],cwd=ROOT,check=True)
    value=compile_evidence(args.implementation_head)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'tests':value['tests'],'recommendation':value['recommendation'],'references':len(value['references'])},indent=2))
