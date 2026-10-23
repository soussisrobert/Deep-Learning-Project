"""Preparation-only Wave5 inventories and unapproved proposals. Never launches training."""
import hashlib
import json
from pathlib import Path
import subprocess

W4_HEAD='b8b92a7c1a1420925dc4b243c09babac3381b66b'
CONFIG_SHA='420de51fc52136043cb05d30a1a203f6abc862de6a80aef60dec0f9b7a5bae6e'
BLOCKERS={
    'W5-B1':'No inclusive external900second/expiry watchdog or durable cumulative failed-work ledger in frozen CLI',
    'W5-B2':'No enforced one-use experiment/command/output/source-closure grant binding or actual GPU lease in frozen CLI',
    'W5-B3':'No host-memory guard; CUDA memory checked after update; finite gradients checked after whole window',
    'W5-B4':'Malformed/nonfinite/mask failures can be ordinary rejected samples rather than fatal contract violations',
    'W5-B5':'No live frozen-backbone/BN/parameter-scope or full source-drift invariant checks',
    'W5-B6':'No durable per-component/per-class/denominator/host-memory/failure-last-checkpoint telemetry',
    'W5-B7':'Physical GPU/host headroom must be rechecked and made safe; no unrelated processes may be killed by this preparation',
}


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def exclusive_json(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False,sort_keys=True);stream.write('\n')


def verify_map(mapping):
    for path,expected in mapping.items():
        if digest(path)!=expected:raise ValueError('frozen content mismatch:'+str(path))
    return len(mapping)


def source_inventory(root):
    root=Path(root).resolve()
    names=subprocess.check_output(['git','ls-files','--','src','scripts','configs','requirements','pyproject.toml','.gitattributes'],cwd=root,text=True).splitlines()
    names+=['src/hift_research/wave5_preparation.py','scripts/prepare_wave5.py','tests/test_wave5_preparation.py']
    upstream=subprocess.check_output(['git','ls-files'],cwd=root/'third_party/HiFT',text=True).splitlines()
    names+=['third_party/HiFT/'+name for name in upstream if Path(name).suffix in ('.py','.yaml','.yml','.json')]
    return {name:digest(root/name) for name in sorted(set(names))}


def authentic_asset_inventory(root):
    root=Path(root).resolve();manifest=read_json(root/'artifacts/wave4-source/SOURCE-001/temporal_manifest.json')
    files={};frames=0;archives=0
    for video in manifest['videos']:
        files[video['source_annotation_path']]=video['source_annotation_sha256']
        files[video['original_seqinfo']['path']]=video['original_seqinfo']['sha256']
        for frame in video['frames']:
            files[frame['path']]=frame['sha256'];frames+=1
    for split in ('train','development'):
        ledger=root/'artifacts/wave4-preflight/PREFLIGHT-002'/split/'candidate_ledger.jsonl'
        with ledger.open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line)
                if row['status']=='accepted':
                    files[row['array_archive_path']]=row['array_archive_sha256'];archives+=1
    verify_map(files)
    if frames!=160 or archives!=1151:raise ValueError('authentic asset accounting changed')
    return {'source_frames':frames,'accepted_archives':archives,'file_sha256':files,
            'bytes':sum(Path(path).stat().st_size for path in files)}


def initialization_identity(root):
    """CPU construction only: source backbone + random grader, no images/loss/optimizer."""
    import torch
    from .training.graph import B2TrainingGraph
    from .training.recipe import load_legacy_backbone_strict
    from .training.lifecycle import apply_epoch_modes
    from .training.wave4_lifecycle import seed_device
    if torch.cuda.is_initialized():raise RuntimeError('preparation process must not initialize CUDA')
    seed_device(20261023,'cpu');model=B2TrainingGraph()
    receipt=load_legacy_backbone_strict(model.backbone,Path(root)/'third_party/HiFT/pretrained_models/alexnet-bn.pth')
    apply_epoch_modes(model,0)
    def values(items):
        result=hashlib.sha256()
        for name,value in items:
            result.update(name.encode());result.update(value.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()
    parameters=values(model.named_parameters())
    previous=read_json(Path(root)/'artifacts/wave4-feasibility/REAL-FORWARD-002/manifest.json')
    if parameters!=previous['parameters_sha256_before']:raise ValueError('seeded initial parameters differ from actual Wave4 forward model')
    result={'all_parameters_sha256':parameters,'backbone_state_sha256':values(model.backbone.state_dict().items()),
        'grader_parameters_sha256':values(model.grader.named_parameters()),'source_load':receipt,
        'cuda_initialized':torch.cuda.is_initialized(),'images_loaded':0,'forward_calls':0,'backward_calls':0,'optimizer_steps':0,
        'seed':20261023,'method':'CPU seed_device; B2TrainingGraph; strict five-counter sourcebackbone load; epoch0 modes'}
    if result['cuda_initialized']:raise RuntimeError('unexpected CUDA initialization')
    return result


def unapproved_proposal(root,experiment_id,inventory_ref,expires_utc,gpu_uuid):
    root=Path(root).resolve();output=root/'artifacts/b2-pilot'/experiment_id
    authorization=root/'artifacts/wave5-authorization'/(experiment_id+'.approved.json')
    if output.exists() or authorization.exists():raise FileExistsError('experiment output or approval already exists')
    argv=[str(root/'.venv/Scripts/python.exe'),str(root/'scripts/run_b2_pilot.py'),'--config',str(root/'configs/B2-Pilot-W4.json'),
        '--authorization-file',str(authorization),'--ack-real-sgd','--output',str(output)]
    return {'schema':'hift-explicit-future-pilot-authorization-v1','proposal_status':'BLOCKED_UNAPPROVED_NOT_EXECUTABLE',
        'execution_ready':False,'human_approved':False,'separate_post_wave4_request':False,'allow_real_sgd':False,
        'human_request_reference':None,'exclusive_gpu_token':None,'experiment_id':experiment_id,
        'wave4_handoff':W4_HEAD,'config_sha256':CONFIG_SHA,'cwd':str(root),'argv':argv,
        'approved_grant_path':str(authorization),'output_path':str(output),'inventory_ref':inventory_ref,
        'device':'cuda:0','gpu_uuid':gpu_uuid,'exclusive_gpu_required':True,'expires_utc':expires_utc,
        'max_effective_updates':100,'max_training_attempts':1600,'max_windows':100,'max_wall_seconds':900,
        'seed':20261023,'source_epoch':0,'dtype':'float32','tf32':False,
        'trainable_scope':'entire corrected grader fusion/transformer/prediction heads; backbone and backboneBN frozen',
        'initialization':'strict source ImageNet backbone plus randomgrader; seeded initial tensor hashes in inventory; no released tracker grader',
        'optimizer':{'kind':'SGD','grader_lr':.0025,'momentum':.9,'weight_decay':.0001,'clip_norm':10},
        'batch':{'attempts_per_micro':4,'attempts_per_window':16,'max_microbatches':4,'refill':False,'observed_global_denominators':True},
        'development':{'updates':[0,25,50,100],'attempts_per_check':128,'positive':64,'negative':64,
            'negative_unique_pairs':38,'negative_repeat_draws':26,'all_pair_draws_unique':False,
            'primary':'fixedlast100','secondary':'bestdev earliesttie; cannot rescue primary',
            'signal':'last100 <=0.95*actualrun initial0; no tracking-performance interpretation'},
        'memory_limits_proposed_not_approved':{'host_process_peak_rss_bytes':8*1024**3,'minimum_available_host_bytes':4*1024**3,
            'max_cuda_reserved_fraction':.8,'require_human_review':True},
        'wall_accounting':'external cumulative monotonic900s includes initialization/input/backward/DEV/checkpoint/failedwork; deadline also bounded by UTCexpiry',
        'max_invocations':1,'resume_allowed':False,'automatic_retry':False,'scope_expansion':False,
        'mandatory_unresolved_launch_blockers':BLOCKERS,
        'approval_rule':'Do NOT approve execution of this blocked draft. Additive safeguard implementation changes executable inventory and requires a NEW complete freeze/proposal/hash/human review. Never mutate this draft or synthesize human evidence.',
        'claims_forbidden':['fullHiFTreproduction','trackingaccuracygain','causalcorrectiongain','independentN128DEV','R1support','additionaltrainingauthority']}


def assert_unapproved(proposal):
    for name in ('human_approved','separate_post_wave4_request','allow_real_sgd','execution_ready'):
        if proposal.get(name) is not False:raise PermissionError('preparation must remain explicitly unapproved:'+name)
    if proposal.get('human_request_reference') is not None or proposal.get('exclusive_gpu_token') is not None:
        raise PermissionError('cannot manufacture human approval or GPU ownership')
    if (proposal.get('max_effective_updates'),proposal.get('max_training_attempts'),proposal.get('max_wall_seconds'))!=(100,1600,900):
        raise ValueError('frozen pilot budget changed')
    if proposal.get('resume_allowed') is not False or proposal.get('automatic_retry') is not False:
        raise PermissionError('recovery requires renewed separate authorization')
    if not proposal.get('mandatory_unresolved_launch_blockers'):raise ValueError('do not erase known mandatory blockers')
    return True
