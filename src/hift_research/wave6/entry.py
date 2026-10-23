"""The sole normal project entry for a future exact authorized B2 pilot (stdlib)."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
from .authorization import read_acknowledgment,verify_bound_files,reverify_sources,claim_authorization,file_digest
from .lease import acquire_lease,probe_physical_availability
from .ledger import reserve_attempt
from .supervisor import supervise,parent_budget_guard


def _exclusive(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def worker_launch_profile(grant,actual_argv):
    """Avoid Windows venv redirector without weakening strict child/lease PPID proof."""
    source_root=Path(actual_argv[1]).resolve().parents[1]
    profile={'application':str(Path(sys._base_executable).resolve()),'logical_python':str(Path(sys.executable).resolve()),
        'venv_launcher':str(Path(sys.executable).resolve()),'python_prefix':str(Path(sys.prefix).resolve()),
        'CUDA_VISIBLE_DEVICES':'' if grant.synthetic_only else grant.gpu_uuid,
        'PYTHONPATH':str(source_root/'src'),'PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'bytecode_cache_policy':'verified_project_source_loader_no_bytecode','interpreter_optimization':0,
        'PYTHONOPTIMIZE':'0','PYTHONHOME':'',
        'argv':[actual_argv[0],actual_argv[1],'--_worker',str(Path(grant.output_path)/'dispatch.json')]}
    if not grant.synthetic_only:
        if grant.proposal['scope'].get('worker_launch')!=profile:raise PermissionError('unreviewed physical/logical worker launch profile')
        initial=grant.proposal['initialization_files_sha256']
        for name in ('application','logical_python'):
            path=profile[name]
            if initial.get(path)!=file_digest(path):raise PermissionError('worker interpreter bytes absent from exact initialization binding')
    return profile


def validate_worker_receipt(result,grant):
    """A zero process exit is not a successful guarded experimental protocol."""
    telemetry=result.get('telemetry',{});completed=telemetry.get('last_events',{}).get('worker_completed')
    checkpoint=result.get('last_checkpoint')
    if result.get('status')=='COMPLETED':
        if (telemetry.get('integrity')!='valid_prefix' or telemetry.get('incomplete_tail') or not completed
                or telemetry.get('last_events',{}).get('failure') or not checkpoint
                or completed.get('updates')!=grant.proposal['limits']['max_effective_updates']
                or not isinstance(completed.get('attempts'),int) or completed['attempts']>grant.proposal['limits']['max_training_attempts']):
            result['status']='FAIL_WORKER_PROTOCOL';result['reason']='incomplete_or_invalid_durable_worker_completion'
    if checkpoint:
        path=Path(checkpoint['path']).resolve();output=Path(grant.output_path).resolve()
        if output not in path.parents or not path.is_file() or file_digest(path)!=checkpoint['sha256']:
            result['status']='FAIL_CHECKPOINT';result['reason']='last_checkpoint_bytes_or_bound_path_invalid'
        else:
            result['checkpoint_bytes_revalidated']=True;telemetry['checkpoint_bytes_revalidated']=True
    return result


def run_guarded(proposal_path,authorization_path,output,*,actual_argv,starting_monotonic=None,allow_synthetic=False):
    origin=time.perf_counter() if starting_monotonic is None else starting_monotonic
    proposal=json.loads(Path(proposal_path).read_text(encoding='utf-8'))
    grant=read_acknowledgment(proposal_path,authorization_path,actual_argv=actual_argv,actual_cwd=str(Path.cwd().resolve()),
        actual_output=str(Path(output).resolve()),gpu_uuid=proposal['device']['gpu_uuid'],allow_synthetic=allow_synthetic)
    if bool(allow_synthetic)!=grant.synthetic_only:raise PermissionError('synthetic-test command is CPU/random fixtures only, never authentic execution')
    directory=Path(grant.proposal['claim_directory']).resolve()
    ledger_path=(directory/(grant.experiment_id+'.time.jsonl')).resolve()
    if ledger_path.parent!=directory:raise PermissionError('experiment ID cannot escape exact claim directory')
    directory.mkdir(parents=True,exist_ok=True)
    margin=min(5,grant.proposal['limits']['max_wall_seconds']/5)
    ledger=reserve_attempt(ledger_path,experiment_id=grant.experiment_id,
        authorization_id=grant.authorization_digest,max_seconds=grant.proposal['limits']['max_wall_seconds'],
        starting_monotonic=origin,expiry_utc=grant.proposal['expires_utc'],cleanup_margin_seconds=margin)
    # External to all training code; even parent-side preflight/lease/cleanup hangs are bounded.
    with parent_budget_guard(ledger):
        lease=None;result=None;claimed=False;released=False;cleanup_error=None;output=Path(output).resolve()
        try:
            verify_bound_files(grant,verify_assets=True);profile=worker_launch_profile(grant,actual_argv)
            ledger.record('verified_inputs',proposal_sha256=grant.proposal_sha256)
            claim_authorization(grant);claimed=True;output.mkdir(parents=True,exist_ok=False)
            if not grant.synthetic_only:
                availability=probe_physical_availability(grant.gpu_uuid,grant.proposal['resources'])
                ledger.record('physical_availability',availability=availability)
                if not availability.get('eligible'):raise PermissionError('physical GPU/host availability not established:'+str(availability))
                lease=acquire_lease(grant.gpu_uuid,grant.authorization_digest,resources=grant.proposal['resources']);lease.assert_owned()
            reverify_sources(grant);capability=secrets.token_hex(32)
            dispatch={'proposal_path':str(grant.proposal_path),'approval_path':str(grant.approval_path),
                'parent_argv':list(actual_argv),'parent_pid':os.getpid(),'synthetic_only':grant.synthetic_only,
                'authorization_digest':grant.authorization_digest,'proposal_sha256':grant.proposal_sha256,
                'capability_sha256':hashlib.sha256(capability.encode()).hexdigest(),'worker_launch':profile,
                'starting_monotonic':origin,'soft_deadline_monotonic':ledger.deadline-margin,
                'lease_path':str(lease.path) if lease else None,'lease_token':lease.token if lease else None}
            dispatch_path=output/'dispatch.json';_exclusive(dispatch_path,dispatch)
            env=dict(os.environ);env['HIFT_W6_CAPABILITY']=capability
            for key in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUBLAS_WORKSPACE_CONFIG','CUDA_VISIBLE_DEVICES','PYTHONOPTIMIZE','PYTHONHOME'):env[key]=profile[key]
            env['__PYVENV_LAUNCHER__']=profile['venv_launcher']
            result=supervise(profile['argv'],cwd=grant.proposal['cwd'],output=output,experiment_id=grant.experiment_id,
                authorization_id=grant.authorization_digest,max_seconds=grant.proposal['limits']['max_wall_seconds'],
                expiry_utc=grant.proposal['expires_utc'],starting_monotonic=origin,ledger=ledger,cleanup_margin_seconds=margin,
                telemetry_path=output/'telemetry.jsonl',env=env,physical_executable=profile['application'],expected_prefix=profile['python_prefix'])
            validate_worker_receipt(result,grant)
            return result
        except BaseException as exc:
            if result is not None:
                result['status']='FAIL_ENTRY';result['reason']=type(exc).__name__+':'+str(exc)
            ledger.record('entry_failure',error=type(exc).__name__+':'+str(exc),claim_consumed=claimed,
                last_committed=None if result is None else result.get('last_committed'),automatic_retry=False)
            raise
        finally:
            try:
                if lease is not None:
                    # On uncertain containment retain the persistent lease artifact. Kernel ownership
                    # eventually closes with this process, but a new launch must fail stale/uncertain.
                    if result is None or not result.get('process_tree_empty_verified'):
                        cleanup_error='lease retained: process tree emptiness unproved; manual reconciliation required'
                    else:lease.release();released=True
            except BaseException as exc:cleanup_error=type(exc).__name__+':'+str(exc)
            finally:
                status=result['status'] if result is not None else 'ENTRY_FAILED'
                if cleanup_error:status='FAIL_LEASE_CLEANUP'
                receipt_path=None
                if result is not None:
                    result['lease_released']=released;result['lease_cleanup_error']=cleanup_error
                    result['ledger_pending_entry_finalization']=True;result['final_ledger_path']=str(ledger.path)
                    receipt_path=output/'guarded_entry_receipt.json'
                    try:_exclusive(receipt_path,result)
                    except BaseException as exc:
                        status='FAIL_FINAL_RECEIPT';cleanup_error=type(exc).__name__+':'+str(exc)
                # Authoritative final ledger is AFTER receipt publication AND lease cleanup.
                final=ledger.finish(status,worker_receipt=result,lease_released=released,lease_cleanup_error=cleanup_error,
                    synthetic_only=grant.synthetic_only,claim_consumed=claimed,entry_receipt_path=str(receipt_path) if receipt_path else None)
                if result is not None:
                    result['ledger_finalization']=final;result['ledger_pending_entry_finalization']=False
                    result['status']=final['status'];result['final_timing']=final['timing']
            if cleanup_error:raise RuntimeError(cleanup_error)
