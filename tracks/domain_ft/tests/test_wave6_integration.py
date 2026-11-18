"""Actual unchanged W4 core + W6 hooks; RANDOM CPU tensors only, no real SGD."""
import copy
import hashlib
import json
import os
import shutil
from datetime import datetime,timedelta,timezone
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch

from hift_research.training.graph import B2TrainingGraph
from hift_research.training.wave4_pilot_engine import SerialPilotController
from hift_research.training.wave4_loader import SyntheticCandidateDataset,dataset_identity
from hift_research.training.wave4_lifecycle import seed_device,device_rng
from hift_research.wave6.adapter import GuardedController
from hift_research.wave6.telemetry import DurableJournal,read_last_good
from hift_research.wave6.runtime import GuardViolation

ROOT=Path(__file__).resolve().parents[1]
SAFE={'host_rss_bytes':1024**3,'available_host_bytes':16*1024**3,'allocated_bytes':0,'reserved_bytes':0,'total_bytes':16*1024**3,'device':'cpu'}


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def make(tmp_path,*,guarded=True,fault=None,rejected=(),probe=None,length=32):
    seed_device(20261023,'cpu');dataset=SyntheticCandidateDataset(length=length,rejected=rejected)
    identity={'kind':'synthetic',**dataset_identity(dataset)}
    kwargs={'config_identity':{'fixture':'W6actual_core'},'data_identity':identity,'synthetic':True,
        'physical':4,'attempts_per_window':16,'max_updates':2,'max_attempts':length}
    if not guarded:return SerialPilotController(B2TrainingGraph(),dataset,**kwargs)
    return GuardedController(B2TrainingGraph(),dataset,journal=DurableJournal(tmp_path/'events.jsonl'),
        resource_probe=probe or (lambda:dict(SAFE)),fault=fault,**kwargs)


def events(controller):return read_last_good(controller.journal.path)['events']


def test_actual_guarded_core_pixel_math_optimizer_rng_bitexact_to_frozen_cpu_core(tmp_path):
    baseline=make(tmp_path/'baseline',guarded=False);baseline.run_to_end()
    state=copy.deepcopy(baseline.model.state_dict());optimizer=copy.deepcopy(baseline.optimizer.state_dict());rng=device_rng(torch.device('cpu'))
    guarded=make(tmp_path/'guarded');guarded.run_to_end()
    assert equal(state,guarded.model.state_dict())
    assert equal(optimizer,guarded.optimizer.state_dict())
    assert equal(rng,device_rng(torch.device('cpu')))
    assert equal(baseline.semantic_trace,guarded.semantic_trace)
    rows=events(guarded)
    assert len([r for r in rows if r['event']=='update_commit'])==2
    assert len([r for r in rows if r['event']=='micro_backward'])==8
    assert all('components' in r and 'denominators' in r for r in rows if r['event']=='loss')
    assert guarded.guard.initial_snapshot['backbone']==guarded.guard.snapshot()['backbone']


@pytest.mark.parametrize('fault',['nan_gradient','backbone','trainable_scope','norm_overflow','host_memory','cuda_memory','crash_midwindow'])
def test_actual_integration_failure_before_next_optimizer_commit(tmp_path,fault):
    resource=dict(SAFE)
    def inject(stage,c):
        if stage=='after_backward':
            if fault=='nan_gradient':next(p for p in c.model.grader.parameters() if p.grad is not None).grad.view(-1)[0]=float('nan')
            if fault=='norm_overflow':next(p for p in c.model.grader.parameters() if p.grad is not None).grad.fill_(1e38)
            if fault=='backbone':
                with torch.no_grad():next(c.model.backbone.parameters()).add_(1)
            if fault=='trainable_scope':next(c.model.backbone.parameters()).requires_grad_(True)
            if fault=='crash_midwindow':raise RuntimeError('injected after backward before SGD')
        if stage=='pre_step':
            if fault=='host_memory':resource['host_rss_bytes']=9*1024**3
            if fault=='cuda_memory':resource.update(reserved_bytes=15*1024**3,allocated_bytes=14*1024**3)
    controller=make(tmp_path,fault=inject,probe=lambda:resource)
    initial=copy.deepcopy(dict(controller.model.grader.named_parameters()))
    with pytest.raises((GuardViolation,RuntimeError)):controller.step()
    assert controller.update==0 and controller.cursor==0 and controller.last_published_update==0 and controller.tainted
    assert not controller.optimizer.state
    assert all(torch.equal(initial[n],p) for n,p in controller.model.grader.named_parameters())
    assert not [r for r in events(controller) if r['event']=='update_commit']
    with pytest.raises(RuntimeError,match='tainted'):controller.step()
    with pytest.raises(RuntimeError,match='tainted'):controller.save_boundary(tmp_path/'invalid.pt')


def test_poststep_failure_is_tainted_never_published_not_false_atomic_rollback(tmp_path):
    def inject(stage,c):
        if stage=='post_step':
            with torch.no_grad():next(c.model.backbone.parameters()).add_(1)
    controller=make(tmp_path,fault=inject)
    with pytest.raises(GuardViolation):controller.step()
    assert controller.tainted and controller.update==0 and controller.last_published_update==0
    rows=events(controller)
    assert [r for r in rows if r['event']=='optimizer_step_begin']
    assert not [r for r in rows if r['event']=='update_commit']
    assert rows[-1]['optimizer_may_have_mutated'] is True


def test_all_reject_consumes_slots_without_sgd_or_refill(tmp_path):
    controller=make(tmp_path,rejected=tuple(range(16)),length=16)
    result=controller.step()
    assert controller.cursor==16 and controller.update==0 and controller.window==1
    assert result['semantic']['rejected']==16 and result['semantic']['accepted']==0
    assert not controller.optimizer.state
    assert not [r for r in events(controller) if r['event']=='optimizer_step_begin']
    assert controller.step() is None


def test_partial_geometric_rejection_no_refill_attempts_and_class_telemetry(tmp_path):
    controller=make(tmp_path,rejected=(1,),length=16);controller.step()
    record=next(r for r in events(controller) if r['event']=='attempt_window')
    assert controller.cursor==16 and controller.update==1
    assert record['accepted_positive']+record['accepted_negative']==15
    assert record['requested_positive']+record['requested_negative']==16
    assert record['rejection_reasons']==[{'index':1,'reason':'synthetic_geometry_reject'}]
    assert record['replacements']==0


def test_interrupted_commit_logging_marks_unpublished_step_tainted(tmp_path,monkeypatch):
    controller=make(tmp_path);original=controller.journal.emit
    def emit(event,**fields):
        if event=='update_commit':raise OSError('injected interrupted output write')
        return original(event,**fields)
    monkeypatch.setattr(controller.journal,'emit',emit)
    with pytest.raises(OSError):controller.step()
    assert controller.tainted and controller.last_published_update==0
    assert not [r for r in events(controller) if r['event']=='update_commit']
    assert events(controller)[-1]['optimizer_may_have_mutated']


def test_checkpoint_write_failure_preserves_previous_atomic_bytes(tmp_path,monkeypatch):
    from hift_research.training import wave4_pilot_engine as engine
    controller=make(tmp_path);prior=controller.save_boundary(tmp_path/'prior.pt');before=Path(prior['path']).read_bytes()
    controller.step()
    def interrupted(value,stream):stream.write(b'PARTIAL_SYNTHETIC_CHECKPOINT');raise OSError('injected checkpoint failure')
    monkeypatch.setattr(engine.torch,'save',interrupted)
    with pytest.raises(OSError):controller.save_boundary(tmp_path/'new.pt')
    assert Path(prior['path']).read_bytes()==before and not (tmp_path/'new.pt').exists()
    assert controller.last_checkpoint==prior and controller.tainted
    assert events(controller)[-1]['event']=='checkpoint_failure'


def test_actual_guarded_synthetic_boundary_replay_is_deterministic(tmp_path):
    full=make(tmp_path/'full');full.run_to_end();expected=copy.deepcopy(full.model.state_dict());optimizer=copy.deepcopy(full.optimizer.state_dict());rng=device_rng(torch.device('cpu'))
    partial=make(tmp_path/'partial');partial.step();point=partial.save_boundary(tmp_path/'point.pt')
    resumed=make(tmp_path/'resume');resumed.restore_boundary(point['path'],expected_sha256=point['sha256']);resumed.run_to_end()
    assert equal(expected,resumed.model.state_dict()) and equal(optimizer,resumed.optimizer.state_dict()) and equal(rng,device_rng(torch.device('cpu')))
    assert resumed.last_published_update==2 and resumed.last_published_cursor==32


def test_guarded_cli_missing_real_human_ack_rejects_before_project_import(tmp_path):
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_b2_guarded.py'),'--proposal',str(tmp_path/'no-proposal.json'),
        '--authorization',str(tmp_path/'no-human-ack.json'),'--output',str(tmp_path/'run')],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert result.returncode!=0 and not (tmp_path/'run').exists()


def cli_bundle(directory,*,fault=None):
    from test_wave6_authorization import make_synthetic_bundle,resign_synthetic_bundle
    command=[str(Path(sys.executable).resolve()),str(ROOT/'scripts/run_b2_guarded.py'),'--proposal','{proposal}',
        '--authorization','{authorization}','--output','{output}','--synthetic-test']
    scope={'synthetic_resource_probe':dict(SAFE)}
    if fault:scope.update(fault_stage='after_backward',fault_kind='hang')
    bundle=make_synthetic_bundle(directory,argv=command,scope_extra=scope,
        limits={'max_effective_updates':1,'max_training_attempts':16,'max_windows':1,'max_wall_seconds':12 if fault else 30})
    source_map=bundle['proposal']['source_files_sha256']
    for code_root in (ROOT/'src',ROOT/'scripts',ROOT/'third_party/HiFT'):
        for path in code_root.rglob('*.py'):source_map[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    inventory=Path(bundle['proposal']['bindings']['source_inventory']['path'])
    inventory.write_text(json.dumps({'file_sha256':source_map},sort_keys=True,indent=2)+'\n',encoding='utf-8')
    bundle['proposal']['bindings']['source_inventory']['sha256']=hashlib.sha256(inventory.read_bytes()).hexdigest()
    resign_synthetic_bundle(bundle)
    return bundle


def run_cli(bundle,argv=None):
    root=bundle['namespace'];stdout=root/'stdout.txt';stderr=root/'stderr.txt'
    with stdout.open('wb') as out,stderr.open('wb') as err:
        env=dict(os.environ)
        for key in ('TEMP','TMP','TMPDIR'):env[key]=str(root.parent)
        result=subprocess.run(argv or bundle['proposal']['argv'],cwd=root,env=env,stdout=out,stderr=err,timeout=40)
    # Retain bounded synthetic metadata/receipts (not tensors or real grants) even
    # when TemporaryDirectory deletes its disposable random-CPU fixture afterwards.
    debug=ROOT/'artifacts/wave6-verification/cli-cases'/root.name
    debug.mkdir(parents=True,exist_ok=True);target=debug/f'run-{len(list(debug.glob("run-*")))+1:03d}';target.mkdir()
    for path in [stdout,stderr,*Path(bundle['proposal']['output_path']).glob('*')]:
        if path.is_file() and path.suffix in ('.json','.jsonl','.log','.txt'):shutil.copyfile(path,target/path.name)
    return result,stdout,stderr


def test_actual_guarded_cli_synthetic_success_venv_ppid_commit_checkpoint_and_reuse_denial():
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-cli-') as directory:
        bundle=cli_bundle(directory);result,stdout,stderr=run_cli(bundle)
        assert result.returncode==0,(stderr.read_text(),json.loads(stdout.read_text()).get('status') if stdout.stat().st_size else '',
            (Path(bundle['proposal']['output_path'])/'child.stderr.log').read_text() if (Path(bundle['proposal']['output_path'])/'child.stderr.log').exists() else '')
        receipt=json.loads(stdout.read_text());assert receipt['status']=='COMPLETED'
        assert receipt['process_tree_empty_verified'] and receipt['job_closed'] and receipt['checkpoint_bytes_revalidated']
        assert not receipt['ledger_pending_entry_finalization'] and not receipt['final_timing']['budget_breached']
        assert receipt['last_committed']['update']==1 and receipt['last_committed']['cursor']==16
        assert receipt['expected_prefix']==str(Path(sys.prefix).resolve())
        assert Path(receipt['physical_executable'])==Path(sys._base_executable)
        checkpoint=Path(receipt['last_checkpoint']['path']);before=checkpoint.read_bytes()
        result,_,_=run_cli(bundle)
        assert result.returncode!=0 and checkpoint.read_bytes()==before


@pytest.mark.parametrize('case',['missing_ack','expired','wrong_role','changed_proposal','changed_source','changed_data','changed_init','wrong_output','duplicate_command_flag','legacy_boolean_grant','production_rejects_synthetic'])
def test_actual_guarded_cli_unauthorized_inputs_never_create_worker_output(case):
    from test_wave6_authorization import resign_synthetic_bundle
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-cli-deny-') as directory:
        bundle=cli_bundle(directory);p=bundle['proposal'];argv=list(p['argv'])
        if case=='missing_ack':bundle['approval_path'].rename(bundle['namespace']/'unused-synthetic-ack.json')
        elif case=='expired':
            p['expires_utc']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();resign_synthetic_bundle(bundle)
        elif case=='wrong_role':
            a=bundle['approval'];a['human_ack']['role']='assistant';bundle['approval_path'].write_text(json.dumps(a),encoding='utf-8')
        elif case=='changed_proposal':
            p['scope']['unreviewed_change']=True;bundle['proposal_path'].write_text(json.dumps(p),encoding='utf-8')
        elif case=='changed_source':(bundle['namespace']/'fixture.py').write_text('# changed byte closure\n',encoding='utf-8')
        elif case=='changed_data':Path(next(iter(p['data_files_sha256']))).write_bytes(b'CHANGED_SYNTHETIC_INPUT')
        elif case=='changed_init':Path(next(iter(p['initialization_files_sha256']))).write_bytes(b'CHANGED_SYNTHETIC_INITIALIZATION')
        elif case=='wrong_output':argv[argv.index('--output')+1]=str(bundle['namespace']/'wrong-output')
        elif case=='duplicate_command_flag':argv+=['--output',p['output_path']]
        elif case=='legacy_boolean_grant':bundle['approval_path'].write_text(json.dumps({'human_approved':True,'ack_real_sgd':True,'exclusive_gpu_token':'NOT_AUTHORITY'}),encoding='utf-8')
        elif case=='production_rejects_synthetic':argv.remove('--synthetic-test')
        result,_,_=run_cli(bundle,argv)
        assert result.returncode!=0 and not Path(p['output_path']).exists()
        assert not (bundle['namespace']/'wrong-output').exists()


def test_actual_guarded_cli_hung_real_backward_hook_stops_cpu_tree_and_keeps_initial_checkpoint():
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-cli-hang-') as directory:
        bundle=cli_bundle(directory,fault=True);result,stdout,stderr=run_cli(bundle)
        assert result.returncode!=0
        output=Path(bundle['proposal']['output_path'])
        # Root's independent hard exit may preempt console publication; durable
        # supervisor/ledger/emergency evidence remains authoritative in that case.
        receipts=list(output.glob('*receipt.json'))
        assert receipts,(stderr.read_text(),stdout.read_text())
        receipt=json.loads((output/'supervisor_receipt.json').read_text())
        assert receipt['process_tree_empty_verified'] and receipt['job_closed']
        assert receipt['child_exit_code']!=0 and receipt['status']!='COMPLETED'
        assert receipt['last_checkpoint']['update']==0 and not receipt['last_committed']
        assert receipt['telemetry']['last_events']['attempt_window']['attempt_ids']==list(range(16))
        assert not receipt['telemetry']['last_events'].get('optimizer_step_begin')
        assert Path(receipt['last_checkpoint']['path']).is_file()


def test_cuda_uuid_normalization_uses_device_properties_without_creating_cuda(monkeypatch):
    from types import SimpleNamespace
    from hift_research.wave6.worker import cuda_uuid
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(uuid='GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'))
    assert cuda_uuid('cuda:0')=='GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'


@pytest.mark.parametrize('status',['FAIL_TELEMETRY','FAIL_CONTAINMENT','STOP_DEADLINE','FAIL_WALL_BUDGET_OVERSHOOT','FAIL_WORKER_PROTOCOL','FAIL_CHECKPOINT'])
def test_cli_cannot_advertise_success_for_zero_exit_failed_guard_status(tmp_path,monkeypatch,status):
    from hift_research.wave6 import entry
    spec=importlib.util.spec_from_file_location('wave6_bootstrap_test',ROOT/'scripts/run_b2_guarded.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ack=tmp_path/'synthetic-ack';ack.write_text('test only',encoding='utf-8')
    monkeypatch.setattr(module,'bootstrap_sources',lambda _:{'source_files_sha256':{}})
    monkeypatch.setattr(module.sys,'argv',[str(ROOT/'scripts/run_b2_guarded.py'),'--proposal','unused','--authorization',str(ack),'--output',str(tmp_path/'unused')])
    monkeypatch.setattr(entry,'run_guarded',lambda *a,**k:{'status':status,'child_exit_code':0,'process_tree_empty_verified':True,'job_closed':True,
        'ledger_pending_entry_finalization':False,'ledger_finalization':{'timing':{'budget_breached':False}},'final_timing':{'budget_breached':False}})
    assert module.main()==1


def test_entry_checkpoint_rehash_exception_cannot_finalize_completed(monkeypatch):
    import contextlib
    from types import SimpleNamespace
    from hift_research.wave6 import entry
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-entry-io-') as directory,monkeypatch.context() as patches:
        bundle=cli_bundle(directory);patches.chdir(bundle['namespace'])
        statuses=[]
        class Ledger:
            path=bundle['namespace']/'metadata-only-ledger'
            deadline=10**20
            def record(self,*a,**k):pass
            def finish(self,status,**details):
                statuses.append(status);return {'status':status,'timing':{'budget_breached':False}}
        patches.setattr(entry,'reserve_attempt',lambda *a,**k:Ledger())
        patches.setattr(entry,'parent_budget_guard',lambda _:contextlib.nullcontext())
        patches.setattr(entry,'supervise',lambda *a,**k:{'status':'COMPLETED','child_exit_code':0,'process_tree_empty_verified':True})
        def failed_rehash(*a):raise OSError('injected checkpoint rehash I/O failure')
        patches.setattr(entry,'validate_worker_receipt',failed_rehash)
        with pytest.raises(OSError,match='rehash'):
            entry.run_guarded(bundle['proposal_path'],bundle['approval_path'],bundle['proposal']['output_path'],
                actual_argv=bundle['proposal']['argv'],allow_synthetic=True)
        assert statuses==['FAIL_ENTRY']


def test_verified_source_loader_ignores_stale_same_timestamp_same_size_bytecode(tmp_path,monkeypatch):
    import py_compile
    import importlib.machinery
    source=tmp_path/'src'/'cached_fixture.py';source.parent.mkdir();source.write_bytes(b"VALUE='old'\n")
    stamp=source.stat().st_mtime_ns;py_compile.compile(str(source),doraise=True)
    source.write_bytes(b"VALUE='new'\n");os.utime(source,ns=(stamp,stamp))
    def load(name):
        spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    assert load('before_fresh_policy').VALUE=='old'  # reproducible non-adversarial timestamp cache hazard
    spec=importlib.util.spec_from_file_location('wave6_cache_bootstrap_test',ROOT/'scripts/run_b2_guarded.py')
    bootstrap=importlib.util.module_from_spec(spec);spec.loader.exec_module(bootstrap)
    monkeypatch.setattr(bootstrap,'ROOT',tmp_path)
    old_write=sys.dont_write_bytecode
    original=bootstrap.install_verified_project_loader({str(source):hashlib.sha256(source.read_bytes()).hexdigest()},force=True)
    try:
        assert load('after_verified_source_policy').VALUE=='new'
        source.write_bytes(b"VALUE='bad'\n")
        with pytest.raises(PermissionError,match='source bytes'):load('changed_after_approval')
    finally:importlib.machinery.SourceFileLoader.get_code=original;sys.dont_write_bytecode=old_write


def test_bootstrap_stdlib_import_cannot_execute_sibling_json_shadow(tmp_path):
    scripts=tmp_path/'fixture'/'scripts';scripts.mkdir(parents=True)
    copied=scripts/'run_b2_guarded.py';copied.write_bytes((ROOT/'scripts/run_b2_guarded.py').read_bytes())
    marker=tmp_path/'SHADOW_EXECUTED'
    (scripts/'json.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("BAD")\n',encoding='utf-8')
    result=subprocess.run([sys.executable,str(copied),'--proposal','missing','--authorization','missing','--output','missing'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    assert result.returncode!=0 and not marker.exists()


def test_synthetic_cli_mode_cannot_select_authentic_worker_even_with_mocked_capability(monkeypatch):
    from types import SimpleNamespace
    from hift_research.wave6 import entry
    with tempfile.TemporaryDirectory(prefix='hift-wave6-synthetic-mode-') as directory:
        bundle=cli_bundle(directory)
        monkeypatch.setattr(entry,'read_acknowledgment',lambda *a,**k:SimpleNamespace(synthetic_only=False))
        def forbidden(*a,**k):raise AssertionError('must reject before reserving/launching')
        monkeypatch.setattr(entry,'reserve_attempt',forbidden)
        with pytest.raises(PermissionError,match='CPU/random'):
            entry.run_guarded(bundle['proposal_path'],bundle['approval_path'],bundle['proposal']['output_path'],
                actual_argv=bundle['proposal']['argv'],allow_synthetic=True)


@pytest.mark.parametrize('flags',[['-O'],['-OO'],['-X','dev'],['-B']])
def test_bootstrap_rejects_unbound_interpreter_flags_before_project_import(flags):
    result=subprocess.run([sys.executable,*flags,str(ROOT/'scripts/run_b2_guarded.py'),'--help'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    assert result.returncode!=0


def test_bootstrap_rejects_inherited_optimization_before_project_import():
    env=dict(os.environ);env['PYTHONOPTIMIZE']='1'
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_b2_guarded.py'),'--help'],env=env,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    assert result.returncode!=0


def test_native_worker_environment_explicit_overrides_survive_parent_environment_merge(tmp_path,monkeypatch):
    from hift_research.wave6.windows_job import SuspendedJobProcess
    out,err=tmp_path/'native.out',tmp_path/'native.err'
    monkeypatch.setenv('PYTHONOPTIMIZE','1');monkeypatch.setenv('PYTHONHOME',str(tmp_path/'wrong-home'))
    code='import json,os,sys;print(json.dumps(dict(optimize=sys.flags.optimize,home=os.environ.get("PYTHONHOME"),opt=os.environ.get("PYTHONOPTIMIZE"),prefix=sys.prefix)))'
    env={'PYTHONOPTIMIZE':'0','PYTHONHOME':'','__PYVENV_LAUNCHER__':sys.executable}
    with SuspendedJobProcess([sys.executable,'-c',code],str(ROOT),stdout_path=out,stderr_path=err,
            env=env,physical_executable=sys._base_executable) as child:
        child.resume();assert child.wait_empty(5) and child.poll()==0,err.read_text()
    observed=json.loads(out.read_text());assert observed['optimize']==0 and observed['opt']=='0' and observed['home']==''
    assert Path(observed['prefix'])==Path(sys.prefix)
