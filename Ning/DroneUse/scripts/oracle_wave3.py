"""Explicit offline current-GT frozen-frame crop probes, NEVER ordinary tracking."""
import argparse
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hift_research import OFFICIAL_CHECKPOINT_SHA256
from hift_research.evaluation.assets import read_json,sha256_file,write_json_exclusive
from hift_research.inference import load_original,make_tracker
from hift_research.provenance import source_identity,runtime_identity
from hift_research.geometry import CropGeometry
from hift_research.diagnostics.policy import decision_maps
from hift_research.wave3_diagnostics.analysis import candidate_boxes_in_image,response_features,candidate_ious
from hift_research.wave3_diagnostics.oracle import crop_conditions,emitted_probe_box
from hift_research.wave3_diagnostics.observer_compat import validate_recorded_observer


def logs(path):
    with Path(path).open(encoding='utf-8') as stream: return [json.loads(line) for line in stream]


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--diagnostic-oracle',action='store_true',required=True)
    parser.add_argument('--gpu-token',required=True)
    args=parser.parse_args()
    cases=read_json(args.cases)
    if cases['kind']!='offline_GT_oracle_sampling_manifest_NOT_inference_inputs': raise ValueError('explicit diagnostic case manifest required')
    plan=ROOT/'experiments/wave3_diagnostic_plan.json'
    if cases['plan_sha256']!=sha256_file(plan): raise ValueError('diagnostic plan changed')
    for case in cases['cases']:
        if args.output.resolve().is_relative_to(Path(case['ordinary_B0_directory']).parent.parent.resolve()):
            raise ValueError('oracle output must be separate from ordinary tracking artifacts')
    args.output.mkdir(parents=True,exist_ok=False)
    base={'kind':'offline_current_GT_frozen_frame_probe_NOT_deployable_tracking','status':'running',
          'created_utc':datetime.now(timezone.utc).isoformat(),'source':source_identity(ROOT),
          'cases_sha256':sha256_file(args.cases),'plan_sha256':sha256_file(plan),'gpu_token':args.gpu_token,
          'uses_current_GT':True,'updates_tracker_state':False,'real_data_optimization':False,
          'limitations':['currentGT revealslocation and changeswindow/scale/context priors','translation/side2x2 controls do not isolate learnedfeature quality',
                         'candidate-bestIoU is GT-assisted head-potential not tracking performance','off-image/occluded pixels not recoverable','origin0and1 always both reported']}
    write_json_exclusive(args.output/'started.json',base)
    import cv2
    import torch
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=ROOT/'artifacts/checkpoints/general_model.pth'
    if sha256_file(checkpoint)!=OFFICIAL_CHECKPOINT_SHA256: raise ValueError('checkpoint mismatch')
    config=read_json(ROOT/'configs/B0.json')
    model=load_original(ROOT,checkpoint,ROOT/config['upstream_config'])
    from pysot.core.config import cfg
    base['environment']=runtime_identity()
    base['checkpoint_sha256']=OFFICIAL_CHECKPOINT_SHA256
    base['frozen_model_state_sha256_before']=model_digest(model)
    current_name=None;grids=None;rows=[];max_raw_difference=0.;reproduced=0
    try:
        with torch.no_grad(),(args.output/'probes.jsonl').open('x',encoding='utf-8') as stream:
            for case in cases['cases']:
                name=case['sequence'];index=case['frame']
                if name!=current_name:
                    if grids is not None: grids.close()
                    path=Path(case['dataset_manifest'])
                    if sha256_file(path)!=case['dataset_manifest_sha256']: raise ValueError('dataset manifest changed')
                    dataset=read_json(path)
                    sequence=next(r for r in dataset['sequences'] if r['name']==name)
                    directory=Path(case['ordinary_B0_directory'])
                    for filename,expected in case['ordinary_input_sha256'].items():
                        if sha256_file(directory/filename)!=expected: raise ValueError('ordinary oracle input changed')
                    ordinary=read_json(directory/'manifest.json')
                    if ordinary['status']!='completed' or ordinary['baseline']!='B0' or ordinary['checkpoint']['sha256']!=OFFICIAL_CHECKPOINT_SHA256:
                        raise ValueError('completed official B0 observation required')
                    for critical in ('src/hift_research/inference.py','src/hift_research/geometry.py','src/hift_research/diagnostics/policy.py'):
                        if ordinary['source']['source_files_sha256'][critical]!=sha256_file(ROOT/critical): raise ValueError(f'critical source changed: {critical}')
                    base.setdefault('observer_source_compatibility',{})[name]=validate_recorded_observer(
                        ROOT,directory/'manifest.json',ordinary['source']['source_files_sha256']['scripts/run_sequence.py'])
                    observations=logs(directory/'diagnostics.jsonl')
                    grids=np.load(directory/'policy_grids.npz',allow_pickle=False)
                    initial_path=Path(dataset['root'])/sequence['frame_paths'][0]
                    if sha256_file(initial_path)!=sequence['frame_sha256'][0]: raise ValueError('initial template bytes changed')
                    initial=cv2.imread(str(initial_path),cv2.IMREAD_COLOR)
                    if initial is None: raise ValueError('initial image unavailable')
                    tracker=make_tracker(model,'B0');tracker.init(initial,observations[0]['bbox'])
                    current_name=name
                image_path=Path(dataset['root'])/sequence['frame_paths'][index]
                if sha256_file(image_path)!=sequence['frame_sha256'][index]: raise ValueError('image bytes changed')
                image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
                if image is None: raise ValueError('image unavailable')
                observation=observations[index];geometry=observation['crop_geometry'];prior=observation['prior_state']
                original_gt=np.asarray(sequence['ground_truth'][index],float)
                if not np.isfinite(original_gt).all(): raise ValueError('unavailable GT must not enter oracle')
                for origin in (1,0):
                    gt=original_gt.copy();gt[:2]-=origin
                    conditions=crop_conditions(gt,prior['center'],geometry['source_size'],geometry['ideal_decoder_scale_z'],image.shape)
                    for condition in conditions:
                        record={'sequence':name,'frame':index,'observed_containment_sampling_group':case['observed_containment_group'],
                                'annotation_origin_assumption':origin,**condition}
                        if condition['status']!='ready':
                            rows.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n');continue
                        crop=tracker.get_subwindow(image,np.asarray(condition['center']),cfg.TRACK.INSTANCE_SIZE,condition['source_side'],tracker.channel_average)
                        raw=model.track(crop)
                        boxes=tracker.generate_anchor(raw['loc']).transpose()
                        p=tracker._convert_score(raw['cls1']);z=raw['cls2'].detach().view(-1).cpu().numpy()
                        selected=decision_maps(boxes,p,z,np.asarray(prior['size']),condition['decoder_scale_z'],tracker.window,
                            w2=cfg.TRACK.w2,w3=cfg.TRACK.w3,penalty_k=cfg.TRACK.PENALTY_K,
                            window_influence=cfg.TRACK.WINDOW_INFLUENCE,learning_rate=cfg.TRACK.LR)
                        image_boxes=candidate_boxes_in_image(boxes,condition['center'],condition['decoder_scale_z'])
                        prediction=emitted_probe_box(boxes,selected,prior['size'],condition['center'],condition['decoder_scale_z'],image.shape)
                        record.update(response_features(image_boxes,gt,p,z,selected['selection'],selected['selected_index']))
                        record.update({'probe_bbox':prediction.tolist(),'probe_emitted_iou':float(candidate_ious(prediction[None],gt)[0]),
                                       'selected_index':selected['selected_index'],'interpolation_lr':selected['interpolation_lr']})
                        if condition['condition']=='observed_center_observed_side':
                            actual_geom=CropGeometry.from_tracker(condition['center'],condition['source_side'],cfg.TRACK.INSTANCE_SIZE)
                            if (actual_geom.origin_x,actual_geom.origin_y)!=(geometry['origin_x'],geometry['origin_y']): raise ValueError('observed crop reconstruction mismatch')
                            difference=max(float(np.max(np.abs(value-grids[key][index-1]))) for key,value in
                                (('predicted_crop_boxes',boxes),('class_probability',p),('bce_logits',z)))
                            max_raw_difference=max(max_raw_difference,difference)
                            for key,value in (('predicted_crop_boxes',boxes),('class_probability',p),('bce_logits',z)):
                                np.testing.assert_allclose(value,grids[key][index-1],rtol=1e-5,atol=1e-6,err_msg='observed response must reproduce before oracle acceptance')
                            np.testing.assert_allclose(prediction,observation['bbox'],rtol=1e-6,atol=1e-6)
                            record['observed_raw_max_abs_difference']=difference;reproduced+=1
                        rows.append(record);stream.write(json.dumps(record,allow_nan=False)+'\n')
                print(f'Offline probe {name} frame{index}:4conditions x2origins; no state update',flush=True)
        summary={}
        for origin in (1,0):
            summary[str(origin)]={}
            for group in ('full','partial','outside'):
                summary[str(origin)][group]={}
                for label in ('observed_center_observed_side','GT_center_observed_side','observed_center_enlarged_side','GT_center_enlarged_side'):
                    subset=[r for r in rows if r['annotation_origin_assumption']==origin and r['observed_containment_sampling_group']==group and r['condition']==label and r['status']=='ready']
                    if subset:
                        summary[str(origin)][group][label]={'frame_probes':len(subset),'fully_available':sum(r['fully_available'] for r in subset),
                            'mean_frozen_frame_probe_iou_NOT_tracking_score':float(np.mean([r['probe_emitted_iou'] for r in subset])),
                            'probes_emitted_iou_ge_0_5':sum(r['probe_emitted_iou']>=.5 for r in subset),
                            'probes_any_candidate_iou_ge_0_5':sum(r['best_decoded_candidate_iou']>=.5 for r in subset),
                            'median_nominal_resized_min_side_px':float(np.median([r['nominal_resized_min_side_px'] for r in subset]))}
        base['frozen_model_state_sha256_after']=model_digest(model)
        if base['frozen_model_state_sha256_before']!=base['frozen_model_state_sha256_after']: raise ValueError('frozen model state mutated during oracle probes')
        base.update({'status':'completed','observed_response_reproductions':reproduced,'maximum_observed_raw_difference':max_raw_difference,
            'frame_cases':len(cases['cases']),'probes':len(rows),'budget_exclusions':sum(r['status']!='ready' for r in rows),
            'conditional_probe_summaries_NOT_tracking_metrics':summary,'probes_sha256':sha256_file(args.output/'probes.jsonl')})
        write_json_exclusive(args.output/'manifest.json',base)
    except Exception as error:
        write_json_exclusive(args.output/'failure.json',{'error':repr(error),'partial_rows':len(rows),'ordinary_tracking_unchanged':True})
        raise
    finally:
        if grids is not None: grids.close()


if __name__=='__main__': main()
