"""Bounded authentic production input/CUDA forward-only measurement. NO optimizer/backward."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import cv2
import torch
from hift_research.provenance import sha256,source_identity
from hift_research.evaluation.assets import write_json_exclusive
from hift_research.data_contract.torch_adapter import reject_aware_collate
from hift_research.temporal_contract.pipeline import TemporalPairPipeline
from hift_research.training.graph import B2TrainingGraph
from hift_research.training.lifecycle import actual_heads,apply_epoch_modes,window_denominators,window_loss
from hift_research.training.recipe import ROOT,load_legacy_backbone_strict
from hift_research.training.wave4_lifecycle import seed_device,device_tree
from hift_research.training.wave4_layout import canonicalize_model_inputs
from hift_research.training.wave4_pilot import paired_development_loss
from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
from hift_research.wave4_feasibility import StageObserver,latency


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def windows_memory():
    if sys.platform!='win32':return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                  *[(key,ctypes.c_size_t) for key in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    value=Counters();value.cb=ctypes.sizeof(value)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):return None
    return {key:int(getattr(value,key)) for key in ('WorkingSetSize','PeakWorkingSetSize','PagefileUsage','PeakPagefileUsage')}


def alignment_panel(pipeline,path):
    """Source GT geometry only: first/last frame of each first20-frame window."""
    import numpy as np
    tiles=[];descriptors=[]
    for lineage,video in sorted(pipeline.videos.items()):
        frames=[value for key,value in sorted(video['frames'].items())]
        for frame in (frames[0],frames[19]):
            image=cv2.imread(frame['path'],cv2.IMREAD_COLOR)
            factor=min(640/image.shape[1],400/image.shape[0])
            resized=cv2.resize(image,None,fx=factor,fy=factor)
            tile=np.full((440,640,3),245,dtype=np.uint8)
            tile[40:40+resized.shape[0],:resized.shape[1]]=resized
            for ann in frame['annotations']:
                if ann['mark']!=1 or ann['class_id']!=1 or ann['visibility']<.5:continue
                x,y,w,h=ann['bbox_xywh'];a=(round(x*factor),40+round(y*factor));b=(round((x+w)*factor),40+round((y+h)*factor))
                cv2.rectangle(tile,a,b,(0,210,0),1)
                cv2.putText(tile,ann['track_id'],a,cv2.FONT_HERSHEY_SIMPLEX,.3,(0,120,0),1)
            label=f'{lineage} {video["split"]} frame{frame["frame_id"]} sourceGT ids; x/y-1'
            cv2.putText(tile,label,(5,24),cv2.FONT_HERSHEY_SIMPLEX,.43,(20,20,20),1)
            tiles.append(tile);descriptors.append({'video':lineage,'frame':frame['frame_id'],'source_sha256':frame['sha256']})
    panel=np.concatenate([np.concatenate(tiles[i:i+2],axis=1) for i in range(0,len(tiles),2)],axis=0)
    if not cv2.imwrite(str(path),panel):raise RuntimeError('could not save alignment audit image')
    return {'path':str(path),'sha256':sha256(path),'frames':descriptors,'scope':'visual annotation-alignment audit, NOT tracking predictions or physical-person reidentification'}


def run(args):
    if not args.gpu_token:raise ValueError('explicit exclusive lead GPU token required')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    plan=ROOT/'experiments/wave4_feasibility_plan.json'
    receipt={'schema':'hift-wave4-authentic-real-forward-only-v1','started_utc':datetime.now(timezone.utc).isoformat(),
        'gpu_token':args.gpu_token,'source':source_identity(ROOT),'feasibility_plan_sha256':sha256(plan),
        'manifest_path':str(Path(args.manifest).resolve()),'manifest_sha256':sha256(args.manifest),
        'policy_path':str(Path(args.policy).resolve()),'policy_sha256':sha256(args.policy),
        'config_path':str(Path(args.config).resolve()),'config_sha256':sha256(args.config),
        'real_data_optimizer_steps':0,'real_data_backward_calls':0,'official_tracker_checkpoint_loaded':False,
        'request_indices':list(range(32)),'physical_attempt_slots':4,'replacement_count':0,
        'environment':{'python':sys.version,'torch':str(torch.__version__),'opencv':cv2.__version__,'platform':platform.platform()},
        'claims':'input feasibility and finite forward loss ONLY; no training/convergence/tracking accuracy evidence'}
    write_json_exclusive(output/'manifest.start.json',receipt)
    start=perf_counter();cv2.setNumThreads(1)
    observer=StageObserver()
    pipeline=TemporalPairPipeline(args.manifest,args.policy,stage_observer=observer)
    pipeline_initialization_seconds=perf_counter()-start
    panel=alignment_panel(pipeline,output/'source_gt_alignment.png')
    device=seed_device(config['seed'],'cuda:0',gpu_token=args.gpu_token)
    start=perf_counter()
    model=B2TrainingGraph()
    backbone_path=ROOT/config['source_backbone_path']
    if sha256(backbone_path)!=config['source_backbone_sha256']:raise ValueError('source backbone identity changed')
    backbone_receipt=load_legacy_backbone_strict(model.backbone,backbone_path)
    apply_epoch_modes(model,0);model.to(device)
    torch.cuda.synchronize(device)
    model_initialization_seconds=perf_counter()-start
    before=parameter_digest(model)
    buffers_before={name:value.detach().cpu().clone() for name,value in model.named_buffers()}
    development=None
    if args.development_policy is not None:
        begin=perf_counter()
        dev=TemporalCandidateDataset(args.manifest,args.development_policy,split='development',length=128)
        development=paired_development_loss(model,dev,device)
        torch.cuda.synchronize(device)
        development.update(full_input_forward_seconds=perf_counter()-begin,
            policy_sha256=sha256(args.development_policy),initial_model_no_SGD=True,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if any(not torch.equal(value.detach().cpu(),buffers_before[name]) for name,value in model.named_buffers()):
            raise ValueError('initial development eval changed model buffers')
        del dev;gc.collect()
    torch.cuda.reset_peak_memory_stats(device)
    rows=[];candidate_ms=[];collate_ms=[];transfer_ms=[];forward_ms=[];counts=Counter()
    def forward(candidates,*,representative):
        begin=perf_counter();collated=reject_aware_collate(candidates)
        collated['model_inputs']=canonicalize_model_inputs(collated['model_inputs'])
        collation=(perf_counter()-begin)*1000
        records=[record for record,arrays in candidates]
        row={'indices':[r['candidate_index'] for r in records],
             'accepted':len(collated['accepted_records']),'rejected':len(collated['rejected_records']),
             'negative_accepted':sum(r['candidate_negative'] for r in collated['accepted_records']),
             'representative':representative,'collation_ms':collation,
             'rejection_reasons':[r.get('reason') for r in collated['rejected_records']]}
        inputs=collated['model_inputs']
        if inputs is not None:
            cpu={'template':inputs['template'],'search':inputs['search'],
                 'targets':{key:inputs[key] for key in ('label_cls1','labelxff','labelcls2','weightxff')}}
            denominators=window_denominators([cpu])
            begin=perf_counter();batch=device_tree(cpu,device);torch.cuda.synchronize(device)
            transfer=(perf_counter()-begin)*1000
            begin=perf_counter()
            with torch.no_grad():
                heads=actual_heads(model,batch)
                if not all(bool(torch.isfinite(value).all()) for value in heads):raise ValueError('nonfinite real-data forward head')
                loss=window_loss(heads,batch['targets'],denominators)
            torch.cuda.synchronize(device);elapsed=(perf_counter()-begin)*1000
            row.update(transfer_ms=transfer,forward_loss_ms=elapsed,loss=float(loss),denominators=denominators,
                cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device))
            if representative:transfer_ms.append(transfer);forward_ms.append(elapsed)
            del heads,loss,batch,cpu
        if representative:collate_ms.append(collation)
        return row
    all_start=perf_counter()
    for first in range(0,32,4):
        candidates=[];block_start=perf_counter()
        for index in range(first,first+4):
            begin=perf_counter();candidate=pipeline.candidate(index);candidate_ms.append((perf_counter()-begin)*1000)
            candidates.append(candidate);record=candidate[0]
            counts[('accepted_' if record['status']=='accepted' else 'rejected_')+('negative' if record['candidate_negative'] else 'positive')]+=1
        row=forward(candidates,representative=True)
        row['end_to_end_ms']=(perf_counter()-block_start)*1000;rows.append(row)
        del candidates
    representative_wall_seconds=perf_counter()-all_start
    extra=[]
    if not counts['accepted_negative']:
        if args.preflight_ledger is None:raise ValueError('negative control requires immutable preflight ledger')
        with Path(args.preflight_ledger).open(encoding='utf-8') as stream:
            negative=next((json.loads(line) for line in stream if line.strip() and
                json.loads(line)['status']=='accepted' and json.loads(line)['candidate_negative']),None)
        if negative is None:raise ValueError('preflight has no accepted negative; no control fabrication')
        candidate=pipeline.candidate(negative['candidate_index'])
        expected={key:value for key,value in negative.items() if key not in ('array_archive_path','array_archive_sha256')}
        if candidate[0]!=expected:raise ValueError('negative control differs from frozen ledger')
        extra.append(forward([candidate],representative=False))
    after=parameter_digest(model)
    if before!=after or any(value.grad is not None for value in model.parameters()):
        raise ValueError('real forward-only parameter/gradient invariant failed')
    changed=[name for name,value in model.named_buffers() if not torch.equal(value.detach().cpu(),buffers_before[name])]
    allowed=('.running_mean','.running_var','.num_batches_tracked')
    if any(not name.endswith(allowed) for name in changed):raise ValueError('unexpected non-BN buffer mutation')
    receipt.update(completed_utc=datetime.now(timezone.utc).isoformat(),result='passed_forward_only_no_SGD',
        backbone_load=backbone_receipt,source_GT_alignment_panel=panel,parameters_sha256_before=before,parameters_sha256_after=after,
        parameters_unchanged=True,no_parameter_gradients=True,changed_disposable_BN_buffers=changed,
        BN_policy='train-mode grader BN buffers may update; model discarded, no adapted checkpoint retained',
        pipeline_initialization_seconds=pipeline_initialization_seconds,model_initialization_seconds=model_initialization_seconds,
        counts=dict(counts),representative_wall_seconds=representative_wall_seconds,rows=rows,
        development_initial_forward_only=development,
        shared_layout_bridge_sha256=sha256(ROOT/'src/hift_research/training/wave4_layout.py'),
        GPU_warmup_policy='Initial complete fixed development eval before training forwards, same as intended future pilot; train decoded/crop caches initially empty',
        extra_negative_controls=extra,stage_statistics=observer.summary(),cache_statistics=pipeline.cache_statistics(),
        candidate_ms=candidate_ms,collation_ms=collate_ms,transfer_ms=transfer_ms,forward_loss_ms=forward_ms,
        latency_summary={'candidate':latency(candidate_ms),'collation':latency(collate_ms),
            'H2D':latency(transfer_ms),'forward_loss':latency(forward_ms)},
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device),
        gpu=torch.cuda.get_device_name(device),host_process_memory=windows_memory())
    write_json_exclusive(output/'manifest.json',receipt)
    print(json.dumps({'result':receipt['result'],'counts':receipt['counts'],'candidate_mean_ms':receipt['latency_summary']['candidate']['mean_ms'],
        'forward_mean_ms':receipt['latency_summary']['forward_loss']['mean_ms'],'peak_cuda_MiB':receipt['peak_cuda_allocated_bytes']/2**20},indent=2))
    del model,pipeline;gc.collect();torch.cuda.empty_cache()
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/B2-Pilot-W4.json')
    parser.add_argument('--preflight-ledger',type=Path);parser.add_argument('--development-policy',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-token',required=True)
    run(parser.parse_args())
