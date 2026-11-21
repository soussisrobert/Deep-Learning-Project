"""CPU-only retrospective Wave3 failure attribution; never controls ordinary inference."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hift_research.evaluation.assets import read_json,sha256_file,validate_dataset_manifest,write_json_exclusive
from hift_research.diagnostics.matched import compare_sequence
from hift_research.provenance import source_identity
from hift_research.wave3_diagnostics.analysis import enrich_rows,describe_sequence,select_oracle_frames,cluster_interval,image_quality_proxies


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def quality_comparisons(rows):
    fields=('native_intensity_variance','native_sobel_energy','native_laplacian_variance','target_surround_histogram_distance')
    groups={}
    for r in rows:
        if r['initialization'] or r['ground_truth_unavailable'] or r['joint_input_availability_class']!='full' or r['motion_bin']=='unknown' or r['min10_ideal_iou_upper_bound'] < .5:
            continue
        label='failed' if r['iou']<.1 else ('reliable' if r['iou']>=.5 else None)
        if label is None: continue
        key=(r['source_size_bin'],r['footprint_bin'],r['motion_bin'],r['elongated'])
        groups.setdefault(key,{'failed':[],'reliable':[]})[label].append(r)
    result=[]
    for key,g in groups.items():
        if min(len(g['failed']),len(g['reliable']))<5: continue
        result.append({'stratum':list(key),'failed_frames':len(g['failed']),'reliable_frames':len(g['reliable']),
            'quality_medians':{field:{label:float(np.median([r[field] for r in subset if r.get(field) is not None])) if any(r.get(field) is not None for r in subset) else None
                for label,subset in g.items()} for field in fields},
            'interpretation':'within_sequence_containment_size_footprint_motion_aspect_controlled_exploration_not_causal;time_occlusion_pose_history_remain'})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',nargs=2,action='append',required=True,metavar=('DATASET_MANIFEST','ORDINARY_RUN_ROOT'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import cv2
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    result={'kind':'retrospective_GT_join_NOT_ordinary_inference','created_utc':datetime.now(timezone.utc).isoformat(),
            'source':source_identity(ROOT),'plan_sha256':sha256_file(ROOT/'experiments/wave3_diagnostic_plan.json'),
            'nonpreregistered_secondary_control_sha256':sha256_file(ROOT/'experiments/wave3_diagnostic_addendum.json'),
            'primary_tracking_convention':'native_numeric_unchanged','physical_diagnostic_origins':[1,0],
            'sequences':{},'references':[]}
    oracle_cases=[]
    pooled={str(origin):{baseline:[] for baseline in ('B0','B1-score')} for origin in (1,0)}
    for manifest_name,runs_name in args.panel:
        manifest_path,runs=Path(manifest_name),Path(runs_name)
        dataset=validate_dataset_manifest(read_json(manifest_path))
        result['references'].append({'dataset_manifest':str(manifest_path.resolve()),'sha256':sha256_file(manifest_path),'ordinary_run_root':str(runs.resolve())})
        for sequence in dataset['sequences']:
            name=sequence['name']
            if name in result['sequences']: raise ValueError('duplicate sequence across panels')
            comparison,_,_=compare_sequence(sequence,runs/name/'B0',runs/name/'B1-score',annotation_origin=1,invalid_policy=dataset['invalid_policy'])
            entry={'frames':len(sequence['ground_truth']),'matched_metric_recomputed':True,
                   'primary_success_auc':{b:comparison[b]['metrics']['success_auc'] for b in ('B0','B1-score')},'origins':{}}
            quality={1:{},0:{}}
            gt=np.asarray(sequence['ground_truth'],float)
            for index,relative in enumerate(sequence['frame_paths']):
                path=Path(dataset['root'])/relative
                if sha256_file(path)!=sequence['frame_sha256'][index]: raise ValueError(f'frame bytes changed: {name}/{index}')
                if index==0 or not np.isfinite(gt[index]).all(): continue
                image=cv2.imread(str(path),cv2.IMREAD_COLOR)
                if image is None: raise ValueError(f'cannot decode {path}')
                for origin in (1,0):
                    box=gt[index].copy();box[:2]-=origin
                    quality[origin][index]=image_quality_proxies(image,box)
            for baseline in ('B0','B1-score'):
                directory=runs/name/baseline
                logs=read_rows(directory/'diagnostics.jsonl')
                predictions=np.loadtxt(directory/f'{name}.csv',delimiter=',',ndmin=2)
                with np.load(directory/'policy_grids.npz',allow_pickle=False) as archive:
                    grids={key:archive[key] for key in archive.files}
                for origin in (1,0):
                    rows=enrich_rows(gt,predictions,logs,grids,baseline=baseline,annotation_origin=origin,quality=quality[origin])
                    summary=describe_sequence(rows)
                    summary['matched_quality_strata']=quality_comparisons(rows)
                    entry['origins'].setdefault(str(origin),{})[baseline]=summary
                    pooled[str(origin)][baseline].append(summary)
                    with (args.output/f'{name}--{baseline}--origin{origin}.jsonl').open('x',encoding='utf-8') as stream:
                        for row in rows: stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if baseline=='B0' and origin==1:
                        for case in select_oracle_frames(rows):
                            oracle_cases.append({'sequence':name,**case,'dataset_manifest':str(manifest_path.resolve()),
                                'dataset_manifest_sha256':sha256_file(manifest_path),'ordinary_B0_directory':str(directory.resolve()),
                                'ordinary_input_sha256':{f:sha256_file(directory/f) for f in ('manifest.json','diagnostics.jsonl','policy_grids.npz',f'{name}.csv')}})
            result['sequences'][name]=entry
            print(f'Analyzed {name}: {len(gt)} rows, both baselines, both origin assumptions',flush=True)
    result['panel_summaries']={}
    for origin,baselines in pooled.items():
        result['panel_summaries'][origin]={}
        for baseline,sequences in baselines.items():
            categories={}
            for label in ('full','partial','outside'):
                counts=[s['containment'][label] for s in sequences]
                categories[label]={'frames':sum(c['frames'] for c in counts),'zero_overlap_frames':sum(c['zero_overlap_frames'] for c in counts),
                    'low_iou_frames':sum(c['low_iou_frames'] for c in counts),
                    'sequence_macro_zero_overlap_rate':cluster_interval([c['zero_overlap_frames']/c['frames'] for c in counts if c['frames']])}
            result['panel_summaries'][origin][baseline]={'sequences':len(sequences),'valid_GT_tracking_frames':sum(s['valid_GT'] for s in sequences),
                'unavailable_GT_frames':sum(s['unavailable_GT'] for s in sequences),'containment':categories,
                'zero_overlap_frames':sum(s['zero_overlap_frames'] for s in sequences),
                'failure_not_fully_contained_share_sequence_macro':cluster_interval([s['fraction_zero_overlap_failures_not_fully_contained'] for s in sequences if s['fraction_zero_overlap_failures_not_fully_contained'] is not None]),
                'low_iou_response_contexts':{key:sum(s['low_iou_response_contexts'][key] for s in sequences) for key in sequences[0]['low_iou_response_contexts']},
                'small_source_large_footprint':{key:sum(s['small_source_large_footprint'][key] for s in sequences) for key in sequences[0]['small_source_large_footprint']},
                'source_information_matched_strata_count':sum(len(s['source_information_matched_strata']) for s in sequences),
                'quality_matched_strata_count':sum(len(s['matched_quality_strata']) for s in sequences),
                'min10_excludes_iou_0_5_frames':sum(s['min10_excludes_iou_0_5_frames'] for s in sequences),
                'fully_available_min10_excludes_iou_0_5_frames':sum(s['fully_available_min10_excludes_iou_0_5_frames'] for s in sequences),
                'policy':{key:sum(s['policy'][key] for s in sequences) for key in sequences[0]['policy']},
                'outside_after_unavailable_GT_not_counted_as_observed_transition':sum(s['outside_after_unavailable_GT_not_counted_as_observed_transition'] for s in sequences),
                'source_bbox_not_fully_in_image_frames':sum(s['source_bbox_not_fully_in_image_frames'] for s in sequences),
                'source_bbox_completely_outside_image_frames':sum(s['source_bbox_completely_outside_image_frames'] for s in sequences),
                'full_crop_but_source_bbox_clipped_frames':sum(s['full_crop_but_source_bbox_clipped_frames'] for s in sequences),
                'new_crop_loss_events':sum(s['new_crop_loss_events'] for s in sequences),
                'drift_preceded_crop_loss_events':sum(s['drift_preceded_crop_loss_events'] for s in sequences),
                'good_prediction_before_crop_loss_events':sum(s['good_prediction_before_crop_loss_events'] for s in sequences)}
    result['limitations']=['selected source-video panel, not benchmark-wide or causal inference','bbox containment does not establish optical visibility or absence of occlusion',
        'candidate localization limitation also includes regression/grid/scale; not isolated feature-fusion limitation',
        'current GT used only after ordinary predictions were frozen','source origin unresolved;0/1 sensitivity mandatory',
        'within-sequence matching does not eliminate time/pose/history/background confounding']
    write_json_exclusive(args.output/'summary.json',result)
    write_json_exclusive(args.output/'oracle_cases.json',{'kind':'offline_GT_oracle_sampling_manifest_NOT_inference_inputs',
        'plan_sha256':result['plan_sha256'],'sampling_origin':1,'cases_reused_for_both_origins':True,'cases':oracle_cases})
    print(f'Complete; frozen {len(oracle_cases)} offline oracle cases; no GPU or optimization')


if __name__=='__main__': main()
