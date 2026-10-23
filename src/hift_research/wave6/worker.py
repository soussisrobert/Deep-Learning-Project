"""Private supervised worker; validates parent capability, exact grant and lease first.

Only the guarded entrypoint supplies the inherited capability. Direct invocation
without a real parent/claim/lease is rejected before importing torch/model code.
Synthetic test envelopes cannot select this authentic-data execution branch.
"""
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime,timezone


def file_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def verify_dispatch(dispatch_path):
    from .authorization import verify_authorization,verify_bound_files,assert_claim
    from .lease import verify_parent_lease
    dispatch_path=Path(dispatch_path).resolve();dispatch=json.loads(dispatch_path.read_text(encoding='utf-8'))
    capability=os.environ.get('HIFT_W6_CAPABILITY','')
    if not capability or not hmac.compare_digest(hashlib.sha256(capability.encode()).hexdigest(),dispatch['capability_sha256']):
        raise PermissionError('missing supervised parent capability')
    proposal_path=Path(dispatch['proposal_path']);proposal=json.loads(proposal_path.read_text(encoding='utf-8'))
    grant=verify_authorization(proposal_path,dispatch['approval_path'],actual_argv=dispatch['parent_argv'],
        actual_cwd=str(Path.cwd().resolve()),actual_output=dispatch_path.parent,gpu_uuid=proposal['device']['gpu_uuid'],
        allow_synthetic=bool(dispatch['synthetic_only']),verify_inputs=False)
    verify_bound_files(grant,verify_assets=True)
    import sys
    profile=dispatch['worker_launch']
    if not grant.synthetic_only and profile!=grant.proposal['scope'].get('worker_launch'):
        raise PermissionError('worker launch profile differs from exact human-reviewed scope')
    if (Path(sys.prefix).resolve()!=Path(profile['python_prefix']).resolve()
            or Path(sys.executable).resolve()!=Path(profile['logical_python']).resolve()
            or Path(sys._base_executable).resolve()!=Path(profile['application']).resolve()):
        raise PermissionError('physical interpreter/logical venv identity mismatch before Torch')
    if sys.flags.optimize!=0 or profile.get('interpreter_optimization')!=0:raise PermissionError('unreviewed interpreter optimization')
    for key in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES','PYTHONOPTIMIZE','PYTHONHOME'):
        if os.environ.get(key)!=profile[key]:raise PermissionError('worker environment differs from reviewed profile:'+key)
    import importlib.machinery
    fingerprint=hashlib.sha256(json.dumps(grant.proposal['source_files_sha256'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if (profile.get('bytecode_cache_policy')!='verified_project_source_loader_no_bytecode' or not sys.dont_write_bytecode
            or getattr(importlib.machinery.SourceFileLoader.get_code,'_hift_wave6_source_map_sha256',None)!=fingerprint):
        raise PermissionError('reviewed exact-byte project source loader absent')
    assert_claim(grant,expected_parent_pid=os.getppid())
    if not grant.synthetic_only:
        verify_parent_lease(dispatch['lease_path'],gpu_uuid=grant.gpu_uuid,authorization_digest=grant.authorization_digest,
            expected_parent_pid=os.getppid(),token=dispatch['lease_token'])
    elif not dispatch.get('synthetic_only'):raise PermissionError('synthetic scope mismatch')
    return dispatch,grant


def execute(dispatch_path):
    dispatch,grant=verify_dispatch(dispatch_path)  # deliberately before torch/project ML imports
    if grant.synthetic_only:
        return execute_synthetic(dispatch,grant)
    return execute_authentic(dispatch,grant)


def _source_check(grant,dispatch):
    from .authorization import reverify_sources
    if time.perf_counter()>=dispatch['soft_deadline_monotonic'] or datetime.now(timezone.utc)>=grant.expires_utc:
        raise TimeoutError('inclusive pilot deadline/expiry reached; stop before next optimizer commit')
    reverify_sources(grant)
    # Runtime input assets, not historical preflight arrays, are checked at every safe boundary.
    root=Path(grant.proposal['cwd']).resolve()
    historical_archives=root/'artifacts/wave4-preflight/PREFLIGHT-002'
    for path,expected in grant.proposal['data_files_sha256'].items():
        target=Path(path).resolve()
        if target.suffix=='.npz' and historical_archives in target.parents:continue
        if file_sha(target)!=expected:raise ValueError('live source-data/manifest/policy/reservation content drift:'+path)


def cuda_uuid(device):
    import uuid
    import torch
    value=str(torch.cuda.get_device_properties(device).uuid)
    return 'GPU-'+str(uuid.UUID(value.removeprefix('GPU-')))


def resource_probe(device):
    import ctypes
    from ctypes import wintypes
    import torch
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),
            ('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
            ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
            ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
    class Memory(ctypes.Structure):
        _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD),('ullTotalPhys',ctypes.c_ulonglong),
            ('ullAvailPhys',ctypes.c_ulonglong),('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),
            ('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
    counters=Counters();counters.cb=ctypes.sizeof(counters);memory=Memory();memory.dwLength=ctypes.sizeof(memory)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise OSError('RSS probe failed')
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(memory)):raise OSError('available RAM probe failed')
    result={'host_rss_bytes':int(max(counters.WorkingSetSize,counters.PeakWorkingSetSize)),'available_host_bytes':int(memory.ullAvailPhys),
        'allocated_bytes':0,'reserved_bytes':0,'total_bytes':1}
    if torch.device(device).type=='cuda':
        free,total=torch.cuda.mem_get_info(device)
        result.update(allocated_bytes=torch.cuda.memory_allocated(device),reserved_bytes=torch.cuda.memory_reserved(device),
            total_bytes=total,device_free_bytes=free,device=str(device),device_uuid=cuda_uuid(device))
    return result


def execute_authentic(dispatch,grant):
    import contextlib
    from unittest.mock import patch
    import torch
    from hift_research.training import wave4_pilot as legacy
    from hift_research.training.graph import B2TrainingGraph
    from hift_research.training.lifecycle import apply_epoch_modes
    from hift_research.training.recipe import load_legacy_backbone_strict
    from hift_research.training.wave4_lifecycle import seed_device
    from hift_research.training.wave4_loader import dataset_identity
    from hift_research.training.wave4_recipes import validate_temporal_binding
    from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
    from .adapter import GuardedController,guarded_development
    from .telemetry import DurableJournal
    output=Path(grant.output_path);journal=DurableJournal(output/'telemetry.jsonl')
    root=Path(grant.proposal['cwd']);config_path=Path(grant.proposal['bindings']['config']['path'])
    config=json.loads(config_path.read_text(encoding='utf-8'));controller=None
    journal.emit('worker_start',experiment_id=grant.experiment_id,authorization_digest=grant.authorization_digest,synthetic=False)
    try:
        _source_check(grant,dispatch)
        dataset=validate_temporal_binding(config)
        dev=TemporalCandidateDataset(root/config['data']['manifest'],root/config['data']['development_policy'],split='development',seed=20261024,length=128)
        device=seed_device(config['seed'],'cuda:0',gpu_token=dispatch['lease_token'])
        if cuda_uuid(device).lower()!=grant.gpu_uuid.lower():raise PermissionError('CUDA physical UUID mismatch before model/optimizer construction')
        model=B2TrainingGraph();initialization=load_legacy_backbone_strict(model.backbone,root/config['source_backbone_path']);apply_epoch_modes(model,0)
        value=hashlib.sha256()
        for name,param in model.named_parameters():value.update(name.encode());value.update(param.detach().cpu().contiguous().numpy().tobytes())
        expected_initial=grant.proposal['scope']['initialization']
        def ordered_digest(items):
            result=hashlib.sha256()
            for name,tensor in items:result.update(name.encode());result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return result.hexdigest()
        if (value.hexdigest()!=expected_initial['all_parameters_sha256']
                or ordered_digest(model.backbone.state_dict().items())!=expected_initial['backbone_state_sha256']
                or ordered_digest(model.grader.named_parameters())!=expected_initial['grader_parameters_sha256']):
            raise ValueError('seeded model/BBBN/grader initialization mismatch before optimizer construction')
        identity=dataset_identity(dataset);identity['kind']='authorized_future_real'
        def scoped_bridge(path,authorization_file=None,*,ack_real_sgd=False):
            if Path(path).resolve()!=config_path.resolve() or Path(authorization_file).resolve()!=grant.approval_path.resolve() or not ack_real_sgd:
                raise PermissionError('invalid internal verified-worker legacy bridge')
            _source_check(grant,dispatch)
            return config,{'exclusive_gpu_token':dispatch['lease_token']}
        with patch.object(legacy,'future_authorization',scoped_bridge):
            controller=GuardedController(model,dataset,config_identity={'config_sha256':file_sha(config_path),'guarded_proposal':grant.proposal_sha256},
                data_identity=identity,device=device,gpu_token=dispatch['lease_token'],synthetic=False,physical=4,attempts_per_window=16,
                max_updates=100,max_attempts=1600,config_path=config_path,authorization_file=grant.approval_path,ack_real_sgd=True,
                journal=journal,source_check=lambda:_source_check(grant,dispatch),resource_probe=lambda:resource_probe(device),
                development_identity={**dataset_identity(dev)})
        results=[];checkpoints=[];above=0
        def evaluate_checkpoint():
            nonlocal above
            result=guarded_development(controller,dev);results.append(result)
            baseline=max(results[0]['total_loss'],1e-6);above=above+1 if result['total_loss']>3*baseline else 0
            controller.auxiliary_state={'wave6_proposal_sha256':grant.proposal_sha256,'authorization_digest':grant.authorization_digest,
                'development':results,'inclusive_origin_monotonic':dispatch['starting_monotonic']}
            checkpoints.append(controller.save_boundary(output/f'update-{controller.update:04d}.pt'))
            if above>=2:raise ValueError('two consecutive DEV losses >3x initial')
        evaluate_checkpoint()
        while controller.cursor<1600 and controller.update<100 and controller.window<100:
            result=controller.step(deadline_monotonic=dispatch['soft_deadline_monotonic'])
            if result['semantic']['stepped'] and controller.update in (25,50,100):evaluate_checkpoint()
        if controller.update!=100:raise RuntimeError('completion failure:1600attemptslots/100windows exhausted before100updates')
        final=results[-1]['total_loss'];initial=results[0]['total_loss']
        reliability=final<=3*max(initial,1e-6)
        journal.emit('worker_completed',updates=controller.update,attempts=controller.cursor,reliability=reliability,
            preliminary_signal=final<=.95*initial,primary_update=100,
            secondary_best_update=min(results,key=lambda r:(r['total_loss'],r['update']))['update'],
            final_invariant_check=controller.guard.check('final'),final_snapshot=controller.guard.snapshot(),checkpoints=checkpoints)
        return 0 if reliability else 1
    except BaseException as exc:
        journal.emit('failure',error=type(exc).__name__+':'+str(exc),last_committed_update=controller.last_published_update if controller else 0,
            last_committed_cursor=controller.last_published_cursor if controller else 0,last_checkpoint=controller.last_checkpoint if controller else None,
            live_state_tainted=True,automatic_retry=False,recovery_authorized=False)
        raise


def execute_synthetic(dispatch,grant):
    """Disposable temporary namespace; only the indexed RANDOM synthetic dataset."""
    import torch
    from hift_research.training.graph import B2TrainingGraph
    from hift_research.training.wave4_lifecycle import seed_device
    from hift_research.training.wave4_loader import SyntheticCandidateDataset,dataset_identity
    from .adapter import GuardedController
    from .telemetry import DurableJournal
    output=Path(grant.output_path);journal=DurableJournal(output/'telemetry.jsonl')
    options=grant.proposal['scope'];seed_device(20261023,'cpu');fault_stage=options.get('fault_stage')
    def fault(stage,controller):
        if stage==fault_stage:
            if options.get('fault_kind')=='hang':time.sleep(3600)
            raise RuntimeError('synthetic fault:'+stage)
    dataset=SyntheticCandidateDataset(length=grant.proposal['limits']['max_training_attempts'],rejected=tuple(options.get('rejected',[])))
    identity={'kind':'synthetic',**dataset_identity(dataset)}
    # Explicit counterfactual resource fixtures are confined to this CPU/random-data
    # branch. Authentic execution above ALWAYS uses the native probe, never this key.
    fixture_probe=options.get('synthetic_resource_probe')
    probe=(lambda:dict(fixture_probe)) if fixture_probe is not None else (lambda:resource_probe('cpu'))
    controller=GuardedController(B2TrainingGraph(),dataset,config_identity={'kind':'synthetic_guard_test'},data_identity=identity,
        synthetic=True,max_updates=grant.proposal['limits']['max_effective_updates'],max_attempts=len(dataset),
        journal=journal,source_check=lambda:_source_check(grant,dispatch),resource_probe=probe,fault=fault)
    journal.emit('worker_start',synthetic=True,experiment_id=grant.experiment_id,authorization_digest=grant.authorization_digest,
        resource_probe_kind='explicit_synthetic_counterfactual' if fixture_probe is not None else 'native_windows')
    controller.save_boundary(output/'update-0000.pt')
    while controller.step(deadline_monotonic=dispatch['soft_deadline_monotonic']) is not None:pass
    controller.save_boundary(output/'last.pt')
    journal.emit('worker_completed',synthetic=True,updates=controller.update,attempts=controller.cursor,final_snapshot=controller.guard.snapshot())
    return 0
