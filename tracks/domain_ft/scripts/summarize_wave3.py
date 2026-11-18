"""Compile immutable Wave3 receipts without tracking, model execution, or optimization."""
from pathlib import Path
import json
import sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hift_research.evaluation.assets import read_json,sha256_file,write_json_exclusive


def main():
    refs={
        'wave2_reference':'experiments/wave2_validation.json',
        'handoff_audit':'artifacts/wave3-diagnostics/WAVE3-HANDOFF-001.json',
        'evaluation_plan':'artifacts/wave3-evaluation/PREREG-001/plan.json',
        'evaluation_summary':'artifacts/wave3-evaluation/SUMMARY-001.json',
        'evaluation_audit':'artifacts/wave3-evaluation/CPU-AUDIT-001.json',
        'evaluation_dataset':'artifacts/wave3-evaluation/ACQUISITION-001/dataset_manifest.json',
        'training_acquisition':'artifacts/wave3-training/acquisition-001/acquisition.json',
        'training_manifest':'artifacts/wave3-training/FROZEN-001/training_manifest.json',
        'training_preflight':'artifacts/wave3-training/PREFLIGHT-001/summary.json',
        'training_independent_verification':'artifacts/wave3-training/PREFLIGHT-001/independent_verification.json',
        'training_distributions':'artifacts/wave3-training/DISTRIBUTIONS-001/statistics.json',
        'training_real_negative_edge':'artifacts/wave3-training/EDGE-NEGATIVE-001/result.json',
        'proposed_recipe':'configs/B2-wave3-proposed.json',
        'lifecycle_manifest':'artifacts/wave3-lifecycle/cpu-20261023-b/manifest.json',
        'lifecycle_verification':'artifacts/wave3-lifecycle/cpu-20261023-b/verification.json',
        'retrospective':'artifacts/wave3-diagnostics/RETROSPECTIVE-002/summary.json',
        'retrospective_cases':'artifacts/wave3-diagnostics/RETROSPECTIVE-002/oracle_cases.json',
        'oracle_manifest':'artifacts/wave3-diagnostics/ORACLE-002/manifest.json',
        'oracle_probes':'artifacts/wave3-diagnostics/ORACLE-002/probes.jsonl',
        'diagnostic_plan':'experiments/wave3_diagnostic_plan.json',
        'diagnostic_secondary_addendum':'experiments/wave3_diagnostic_addendum.json'}
    data={key:read_json(ROOT/path) for key,path in refs.items() if not path.endswith('.jsonl')}
    e,d,o,p,l,c,w=(data[key] for key in ('evaluation_summary','retrospective','oracle_manifest','training_preflight','lifecycle_verification','proposed_recipe','wave2_reference'))
    assert p['candidate_count']==1024 and p['accepted_count']+p['rejected_count']==1024
    assert p['replacements']==0 and p['accepted_negative_count']==0 and p['no_model_forward_sgd_gpu']
    assert len(l['comparisons'])==9 and all(l['comparisons'].values())
    assert not c['launch_allowed'] and not c['validated_real_data'] and not c['real_data_sgd_authorized']
    assert c['real_training_manifest_sha256']==p['manifest_sha256']==sha256_file(ROOT/refs['training_manifest'])
    assert sha256_file(ROOT/refs['retrospective_cases'])==o['cases_sha256']
    assert o['maximum_observed_raw_difference']==0 and o['observed_response_reproductions']==192
    assert o['frozen_model_state_sha256_before']==o['frozen_model_state_sha256_after']
    for path,digest in c['manifest_source_code_hashes'].items():
        assert sha256_file(ROOT/path)==digest,path
    sequences=[]
    for row in e['sequences']:
        sequences.append({key:row[key] for key in ('sequence','frames','B0','B1-score')})
    h3=[]
    per_sequence=[]
    for name,row in d['sequences'].items():
        conditions={}
        for origin in ('0','1'):
            conditions[origin]={}
            for baseline in ('B0','B1-score'):
                source=row['origins'][origin][baseline]
                conditions[origin][baseline]={key:source[key] for key in (
                    'containment','unavailable_GT','new_crop_loss_events','drift_preceded_crop_loss_events',
                    'outside_after_unavailable_GT_not_counted_as_observed_transition','policy',
                    'source_bbox_not_fully_in_image_frames','full_crop_but_source_bbox_clipped_frames')}
                if source['source_information_matched_strata'] or source['matched_quality_strata']:
                    h3.append({'sequence':name,'origin':int(origin),'baseline':baseline,
                               'source_size_comparisons':source['source_information_matched_strata'],
                               'quality_comparisons':source['matched_quality_strata']})
        per_sequence.append({'sequence':name,'frames':row['frames'],'conditions':conditions})
    old_metrics={base:{metric:sum(row[base]['metrics'][metric] for row in w['sequences'])/len(w['sequences'])
                       for metric in ('success_auc','precision_at_20px')} for base in ('B0','B1-score')}
    combined={base:{metric:(12*e['macro'][base][metric]+3*old_metrics[base][metric])/15
                    for metric in ('success_auc','precision_at_20px')} for base in ('B0','B1-score')}
    receipt={
        'schema_version':1,'kind':'Wave3_compact_measured_evidence_NOT_training_authorization',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'starting_HEAD':'fba88e5b3452bf6e92980f14a52604d5d93eb0b7',
        'upstream_HEAD':'992aa7c1605360d607af84cb804b93cd8ee81caf',
        'gates':{'W3-G1':'PASS (bounded canonical COCO real-input contract only)',
                 'W3-G2':'PARTIAL','W3-G3':'PASS','W3-G4':'PARTIAL (completed bounded diagnostics; representation vs optical information not isolated)',
                 'W3-G5':'PARTIAL'},
        'hypotheses':{
            'H1':{'assessment':'Supported','scope':'Crop availability is a demonstrated limiting condition in selected failure episodes, not established initiating or universal cause.'},
            'H2':{'assessment':'Plausible but unconfirmed','scope':'Contained low-IoU failures and limited candidate coverage persist; visibility, regression/grid, scale, prior and history are not isolated from representation.'},
            'H3':{'assessment':'Insufficient evidence','scope':'Small native support and weak quality proxies do not isolate source-information causality. No B0 matched quality cell; only one B1 cell in one sequence.'}},
        'exactly_one_primary_next_step':{'choice':2,'recommendation':'Continue data/recipe correctness work',
            'why':'0 accepted negatives; static-only COCO subset; unaccepted four-dataset/loader/BN/accumulation recipe and no real-data SGD authority.',
            'concrete_completion':'Acquire/freeze authentic temporal object-ID data and known-distinct negative pairs, predeclare alias-safe sampling deviations, repeat bounded real-input class/distribution preflight; no real optimization until separately authorized.'},
        'real_training_preflight':p,'training_distributions':data['training_distributions'],
        'training_independent_verification':data['training_independent_verification'],
        'real_negative_branch_control_outside_1024':data['training_real_negative_edge'],
        'lifecycle_comparisons':l['comparisons'],
        'proposed_recipe_launch_allowed':c['launch_allowed'],
        'ordinary_expanded':{'sequences':sequences,'macro':e['macro'],
                             'paired_uncertainty':e['paired_cluster_uncertainty'],
                             'condition_strata':e['condition_strata']},
        'ordinary_Wave2_macro_unchanged':old_metrics,
        'combined_15_sequence_macro_descriptive_NOT_new_runs':combined,
        'retrospective_panel':d['panel_summaries'],
        'retrospective_per_sequence':per_sequence,
        'contained_H3_comparisons':h3,
        'offline_oracle_NOT_ordinary_tracking':{key:o[key] for key in (
            'conditional_probe_summaries_NOT_tracking_metrics','budget_exclusions',
            'observed_response_reproductions','maximum_observed_raw_difference',
            'frozen_model_state_sha256_before','frozen_model_state_sha256_after')},
        'new_network_payload_bytes':{'evaluation':221625794,'training_including_prior_probe':171734757,'combined':393360551},
        'references':{key:{'path':path,'sha256':sha256_file(ROOT/path)} for key,path in refs.items()},
        'limitations':d['limitations']+['training/pretraining leakage is screened within known local reservations, not globally disproven',
             'old/new panels differ; combined means are descriptive, not longitudinal software improvement',
             'oracle same-frame GT controls are conditional diagnostics; no tracking/AUC or controller gain',
             'source-side changes also change scale/context; independent center-by-side controls do not isolate optical information',
             'final corrected CPU retrospective excludes transitions after unknown-GT gaps; its frozen oracle case hash equals the prior completed probe input exactly',
             'R1 and real-data optimization remain prohibited; no novelty claim']}
    write_json_exclusive(ROOT/'experiments/wave3_validation.json',receipt)
    print('Wave3 evidence compiled; exact frozen cases preserved, no model execution')


if __name__=='__main__':main()
